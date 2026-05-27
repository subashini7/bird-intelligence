import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm

# Within this path, <Species_Name> folder has up to 30 images of that species.
DATA_ROOT = '/kaggle/input/datasets/jupiter79/india-birds/processed_singapore_birds/processed_singapore_birds'
OUTPUT_BASE = '/kaggle/working'

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_transforms():
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf

class TransformedSubset(Dataset):
    """Applies a unique transform map to a specific split subset."""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

class DinoBirdClassifier(nn.Module):
    def __init__(self, num_classes, freeze_encoder=False):
        super().__init__()
        self.encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features)

def calculate_topk(logits, labels, k=1):
    with torch.no_grad():
        _, pred = logits.topk(k, dim=1, largest=True, sorted=True)
        correct = pred.eq(labels.view(-1, 1).expand_as(pred))
        return correct.any(dim=1).float().mean().item() * 100.0

def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss, total_top1, total_top5 = 0, 0, 0
    for imgs, labels in tqdm(loader, desc="Training", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            logits = model(imgs)
            loss = criterion(logits, labels)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        total_top1 += calculate_topk(logits, labels, k=1)
        total_top5 += calculate_topk(logits, labels, k=5)
    num_batches = len(loader)
    return total_loss / num_batches, total_top1 / num_batches, total_top5 / num_batches

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_top1, total_top5 = 0, 0, 0
    for imgs, labels in tqdm(loader, desc="Evaluating", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        total_top1 += calculate_topk(logits, labels, k=1)
        total_top5 += calculate_topk(logits, labels, k=5)
    num_batches = len(loader)
    return total_loss / num_batches, total_top1 / num_batches, total_top5 / num_batches

def main():
    parser = argparse.ArgumentParser()
    # Adjusted default path to point to your flat nested Singapore directory path root
    parser.add_argument('--data_dir', default=DATA_ROOT)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--freeze_encoder', action='store_true')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--experiment_name', type=str, default='standalone_model')
    args, unknown = parser.parse_known_args()

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_dir = Path(OUTPUT_BASE) / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load the raw imagery from flat folders without transforms initially
    full_dataset = datasets.ImageFolder(args.data_dir)
    class_names = full_dataset.classes
    print(f"Loaded dataset total: {len(full_dataset)} images spanning {len(class_names)} species.")

    # 2. Split dataset programmatically into train (80%) and val (20%) subsets
    val_size = int(len(full_dataset) * 0.20)
    train_size = len(full_dataset) - val_size
    base_train, base_val = random_split(full_dataset, [train_size, val_size])

    # 3. Apply the appropriate distinct augmentation transforms to each split
    train_tf, val_tf = get_transforms()
    train_ds = TransformedSubset(base_train, transform=train_tf)
    val_ds = TransformedSubset(base_val, transform=val_tf)
    
    kw = dict(batch_size=args.batch_size, num_workers=2, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)

    model = DinoBirdClassifier(num_classes=len(class_names), freeze_encoder=args.freeze_encoder).to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    best_top1 = 0.0
    if args.resume:
        print(f"📂 Loading weights from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("✅ Weights successfully transferred.")

    for epoch in range(args.epochs):
        tr_loss, tr_t1, tr_t5 = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_t1, val_t5 = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Ep {epoch+1:2d}/{args.epochs} | Train Loss: {tr_loss:.4f} Top1: {tr_t1:.1f}% | Val Loss: {val_loss:.4f} Top1: {val_t1:.1f}% | LR: {current_lr:.2e}")
        
        if val_t1 > best_top1:
            best_top1 = val_t1
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'class_names': class_names,
                'best_top1': best_top1
            }
            save_path = f'{OUTPUT_BASE}/checkpoints/{args.experiment_name}_best.pth'
            torch.save(checkpoint_data, save_path)
            print(f"✨ Best checkpoint saved to {save_path}")

if __name__ == '__main__':
    main()
