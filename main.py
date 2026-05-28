import csv
import os
import re
from datetime import date, datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import osxphotos
import pandas as pd
import photoscript
import plotly.express as px
import pyiqa
import reverse_geocoder as rg
import seaborn as sns
import torch
import torch.nn as nn
from Binocular.models.inference import InferenceModel
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from torchvision import transforms
from transformers import pipeline

load_dotenv()
HF_API = os.getenv("HF_TOKEN")
BIRD_TAG_PREFIX = "Bird: "
device = "mps" if torch.backends.mps.is_available() else "cpu"
OUTPUT_FILE = "classified_birds_report.csv"
ALBUM_NAME = "Birds"
DATE_CUTOFF = date(2024, 6, 18)
TARGET_REGION = "India"  # Change to "US", "India", "UK", or "Singapore" as needed
REGION_CONFIG = {
    "US": {
        "country_codes": {"US"},
        "classifier": {
            "repo_id": "jiujiuche/binocular",
            "filename": "artifacts/dinov2_vitb14_nabirds.pth",
            "is_standalone": False,
        },
        "use_scientific_to_common": False,
    },
    "Singapore": {
        "country_codes": {"SG"},
        "classifier": {
            "repo_id": "pshops/dinov2-singapore-birds",
            "filename": "probe_best.pth",
            "is_standalone": True,
        },
        "use_scientific_to_common": True,
        "mapping_csv": "regional_birds.csv",
    },
    "India": {
        "country_codes": {"IN"},
        "classifier": {
            "repo_id": "pshops/dinov2-india-birds",
            "filename": "fine_tune_best.pth",
            "is_standalone": True,
        },
        "use_scientific_to_common": True,
        "mapping_csv": "regional_birds.csv",
    },
    "UK": {
        "country_codes": {"GB"},
        "classifier": {
            "repo_id": "pshops/dinov2-uk-birds",
            "filename": "fine_tune_best.pth",
            "is_standalone": True,
        },
        "use_scientific_to_common": True,
        "mapping_csv": "regional_birds.csv",
    },
}

detector = None
classifier = None
geo_cache = {}


# NOTE: This score is not usable in its current state.
# Water increases the score in an otherwise sharp image.
def get_bird_quality_score(crop_array, iqa_metric):
    """Calculate a no-reference image quality score for a cropped bird region."""
    rgb_img = cv2.cvtColor(crop_array, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)
    # NIQE: Lower is better (typically 0–10)
    score = iqa_metric(pil_img)
    return float(score)


def is_in_target_region(lat, lon):
    """Return True if the given coordinates are within the selected target region."""
    coord_key = (round(lat, 1), round(lon, 1))
    if coord_key not in geo_cache:
        results = rg.search((lat, lon))
        # 'cc' is the ISO country code
        geo_cache[coord_key] = (
            results[0]["cc"] in REGION_CONFIG[TARGET_REGION]["country_codes"]
        )
    return geo_cache[coord_key]


def load_scientific_common_map(csv_filename=None):
    """Load a scientific-to-common name map from regional_birds.csv."""
    if csv_filename is None:
        csv_filename = os.path.join(os.path.dirname(__file__), "regional_birds.csv")
    if not os.path.exists(csv_filename):
        return {}
    try:
        df = pd.read_csv(csv_filename, usecols=["Scientific Name", "Common Name"])
    except Exception:
        return {}
    mapping = {}
    for sci, common in zip(
        df["Scientific Name"].fillna(""), df["Common Name"].fillna("")
    ):
        sci = sci.strip()
        common = common.strip()
        if sci:
            mapping[sci] = common or sci
    return mapping


def get_region_scientific_to_common_map():
    region_cfg = REGION_CONFIG.get(TARGET_REGION, REGION_CONFIG["US"])
    if not region_cfg.get("use_scientific_to_common", False):
        return {}
    csv_filename = region_cfg.get("mapping_csv")
    if csv_filename is None:
        csv_filename = os.path.join(os.path.dirname(__file__), "regional_birds.csv")
    return load_scientific_common_map(csv_filename)


SCIENTIFIC_TO_COMMON = get_region_scientific_to_common_map()


def get_common_name(scientific_name: str) -> str:
    base_name = scientific_name.split(" (")[0].strip()
    return SCIENTIFIC_TO_COMMON.get(
        base_name, SCIENTIFIC_TO_COMMON.get(scientific_name, scientific_name)
    )


def convert_predictions_to_common(predictions):
    return [(get_common_name(label), score) for label, score in predictions]


def make_tag(label: str) -> str:
    """Return a Photos keyword string for a bird label."""
    return f"{BIRD_TAG_PREFIX}{label}"


def is_bird_tag(keyword: str) -> bool:
    """Return True if the keyword was created by this script."""
    return keyword.startswith(BIRD_TAG_PREFIX)


def sync_keywords_from_csv(csv_filename: str):
    """Write refined labels from a CSV back to the macOS Photos app as keywords."""
    df = pd.read_csv(csv_filename)
    # photo.filename in photoscript == original_filename from osxphotos
    label_map = dict(zip(df["display_name"], df["refined_label"].fillna("")))

    photos_app = photoscript.PhotosLibrary()
    photos_app.activate()
    target_album = photos_app.album(ALBUM_NAME)

    if not target_album:
        print(f"Album '{ALBUM_NAME}' not found.")
        return

    updated, cleared, not_found = 0, 0, 0

    for photo in target_album.photos():
        fname = photo.filename
        current_keywords = photo.keywords
        cleaned_keywords = [k for k in current_keywords if not is_bird_tag(k)]

        if fname not in label_map:
            not_found += 1
            continue

        new_label = label_map[fname]

        if new_label:
            photo.keywords = cleaned_keywords + [make_tag(new_label)]
            updated += 1
        else:
            photo.keywords = cleaned_keywords
            cleared += 1

    print(f"Done. Updated: {updated} | Cleared: {cleared} | Not in CSV: {not_found}")


def get_refined_label(predictions, lat, lon, date):
    """
    Determine a refined species label given model predictions and context.

    Applies location-based rules, subspecies grouping, and known mis-classification
    corrections that are beyond the model's training scope.
    """
    # 1. Get the top two labels and scores
    top_label, top_score = predictions[0]
    second_label, second_score = predictions[1]
    sum_top_two = top_score + second_score

    # 2. Extract base names by removing anything in parentheses
    #    e.g. "Yellow-rumped Warbler (Breeding Myrtle)" -> "Yellow-rumped Warbler"
    base_top = top_label.split(" (")[0].strip()
    base_second = second_label.split(" (")[0].strip()

    is_channel_island = (33.0 < lat < 34.5) and (-120.5 < lon < -118.0)
    refined_label = ""

    if top_score > 0.99:
        refined_label = top_label
        if "Western Scrub-Jay" in top_label:
            refined_label = (
                "Island Scrub-Jay" if is_channel_island else "California Scrub-Jay"
            )
        elif top_label == "Green Heron":
            # Green Heron / young Black-crowned Night-Heron are visually ambiguous
            refined_label = "Green Heron or Young Black-crowned Night-Heron"

    # 3. Apply grouping logic when the top two predictions together exceed 99 %
    if sum_top_two > 0.99:
        if base_top == base_second:
            # Combine subspecies or gender splits with the same base name
            # (handles Yellow-rumped Warblers, Harriers, etc.)
            refined_label = base_top
        elif {base_top, base_second} == {"Western Grebe", "Clark's Grebe"}:
            refined_label = "Western/Clark's Grebe"
        elif {base_top, base_second} == {"Common Tern", "Forster's Tern"}:
            refined_label = "Forster's Tern"
        elif {base_top, base_second} == {"Lesser Yellowlegs", "Greater Yellowlegs"}:
            refined_label = "Greater Yellowlegs"
        elif {base_top, base_second} == {"Common Loon", "Pacific Loon"}:
            refined_label = "Common Loon"
        elif any(x in base_top for x in ["Hummingbird", "Gull"]) and any(
            x in base_second for x in ["Hummingbird", "Gull"]
        ):
            # Collapse ambiguous hummingbird / gull pairs to the genus suffix
            refined_label = base_top.split()[-1]

    # 4. Correct known mis-classifications
    # Context: White-headed Woodpecker is absent from training → predicted as Pileated;
    # several other species produce high-confidence false positives on this dataset.
    base_refined_label = refined_label.split(" (")[0].strip()
    corrections = {
        "Tennessee Warbler": "Orange-crowned Warbler",
        "Pileated Woodpecker": "White-headed Woodpecker",
        "Snow Goose": "Ross's Goose",
        "Surfbird": "Dunlin",
        "Downy Woodpecker": "Hairy Woodpecker",
        "Wilson's Phalarope": "Red-necked Phalarope",
    }
    if base_refined_label in corrections:
        refined_label = corrections[base_refined_label]
    elif base_refined_label == "Ring-necked Duck" and date == "2025-02-08":
        refined_label = "Tufted Duck"
    elif base_refined_label == "Bald Eagle" and date == "2025-11-28":
        refined_label = "California Condor"
    elif base_refined_label in {
        "Eastern Bluebird",
        "Gila Woodpecker",
        "Abert's Towhee",
        "American Black Duck",
        "Inca Dove",
        "Black Tern",
    }:
        refined_label = ""

    return refined_label


def detect_classify(photo, is_in_target_region, detector, classifier, iqa_metric):
    """
    Run bird detection and classification on a single photo.

    Returns a dict of results, or None if the photo cannot be read.
    Classification is only attempted for single-bird photos in the target region.
    """
    display_name = photo.original_filename or photo.filename
    location_str = "No GPS Data"
    date_str = photo.date.strftime("%Y-%m-%d %H:%M:%S") if photo.date else "Unknown"
    bird_count = 0
    detect_high_conf = 0
    avg_quality = 0.0
    top_5_summary = ""
    top_label = ""
    top_confidence = 0
    refined_label = ""
    current_label = photo.keywords[0].replace(BIRD_TAG_PREFIX, "") if photo.keywords else ""

    try:
        if (
            not photo.path
            or not os.path.exists(str(photo.path))
            or not is_in_target_region
        ):
            return None
        photo_path = photo.path_edited if photo.path_edited else photo.path
        if isinstance(photo_path, Image.Image):
            image = photo_path
        else:
            image = Image.open(str(photo_path))
        image = image.convert("RGB")
        image.load()  # Force load to catch "premature end of data"

        results = detector(str(photo_path))
        birds = [
            res for res in results if res["label"] == "bird" and res["score"] > 0.7
        ]
        bird_count = len(birds)

        if birds:
            detect_high_conf = birds[0]["score"]
            img_cv = cv2.imread(photo_path)
            if img_cv is None:
                raise ValueError("OpenCV failed to read image")

            scores = []
            loc_data = photo.location

            for b in birds:
                box = b["box"]
                x1, y1, x2, y2 = (
                    int(box["xmin"]),
                    int(box["ymin"]),
                    int(box["xmax"]),
                    int(box["ymax"]),
                )
                if (y2 - y1) > 10 and (x2 - x1) > 10:
                    crop_cv = img_cv[y1:y2, x1:x2]
                    scores.append(get_bird_quality_score(crop_cv, iqa_metric))

                if (
                    (len(birds) == 1)
                    and is_in_target_region
                    and (y2 - y1) > 0
                    and (x2 - x1) > 0
                ):
                    crop_pil = Image.fromarray(cv2.cvtColor(crop_cv, cv2.COLOR_BGR2RGB))
                    predictions = classifier.predict(crop_pil, top_k=5)
                    if REGION_CONFIG.get(TARGET_REGION, {}).get(
                        "use_scientific_to_common", False
                    ):
                        predictions = convert_predictions_to_common(predictions)
                    refined_label = ""
                    top_label, top_confidence = predictions[0]
                    if (
                        REGION_CONFIG.get(TARGET_REGION, {}).get(
                            "use_scientific_to_common", False
                        )
                        and top_confidence > 0.40
                        and top_label not in {
                            "Tawny Eagle",
                            "Vernal Hanging-Parrot",
                            "Marsh Sandpiper",
                        }
                        # "Great White Pelican" is "Indian Spot-billed Pelican" as missing in reference
                        # 8752 - Whimbrel
                        # Great Egret is once classified as Eastern Cattle-Egret
                    ):
                        refined_label = top_label
                    else:
                        refined_label = get_refined_label(
                            predictions,
                            loc_data[0],
                            loc_data[1],
                            photo.date.strftime("%Y-%m-%d"),
                        )
                    pred_list = [
                        f"{label} ({score:.2%})" for label, score in predictions
                    ]
                    top_5_summary = " | ".join(pred_list)

                    print(f"\n--- Predictions for {display_name} ---")
                    for label, score in predictions:
                        print(f"Species: {label:30} | Confidence: {score:.2%}")
                    label_text = refined_label
                    cv2.putText(
                        img_cv,
                        label_text,
                        (x1, y1 - 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (255, 255, 0),
                        3,
                    )

            avg_quality = sum(scores) / len(scores) if scores else 0.0
            if loc_data:
                location_str = f"{loc_data[0]}, {loc_data[1]}"

        print(f"Processed: {display_name} | Found: {bird_count}")
        return {
            "display_name": display_name,
            "location_str": location_str,
            "date_str": date_str,
            "bird_count": bird_count,
            "detect_high_conf": detect_high_conf,
            "avg_quality": round(avg_quality, 3),
            "refined_label": refined_label,
            "current_label": current_label,
            "top_label": top_label,
            "top_confidence": round(top_confidence, 3),
            "top_5_predictions": top_5_summary,
        }
    except Exception as e:
        print(f"Error processing {display_name}: {type(e).__name__}: {e}")
        return None


def process_birds_album(detector, classifier, iqa_metric, csv_filename):
    """Detect and classify every photo in the Birds album and write results to a CSV."""
    db = osxphotos.PhotosDB()
    photos = [
        p
        for p in db.photos()
        if p.path and any(ALBUM_NAME in a.title for a in p.album_info)
    ]
    photos.sort(key=lambda p: getattr(p, "original_filename", "") or "")
    print(f"Found {len(photos)} photos in Birds album. Starting analysis...")

    headers = [
        "display_name",
        "location",
        "date",
        "bird_count",
        "detect_high_conf",
        "avg_q_score",
        "refined_label",
        "current_label",
        "top_label",
        "top_confidence",
        "top_5_predictions",
    ]

    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)

        for photo in photos:
            if getattr(photo, "date", None) and photo.date.date() < DATE_CUTOFF:
                continue
            loc_data = photo.location
            is_in_region = (
                bool(loc_data)
                and loc_data[0] is not None
                and loc_data[1] is not None
                and is_in_target_region(loc_data[0], loc_data[1])
            )
            data = detect_classify(
                photo, is_in_region, detector, classifier, iqa_metric
            )
            if data:
                writer.writerow(
                    [
                        data["display_name"],
                        data["location_str"],
                        data["date_str"],
                        data["bird_count"],
                        data["detect_high_conf"],
                        data["avg_quality"],
                        data["refined_label"],
                        data["current_label"],
                        data["top_label"],
                        data["top_confidence"],
                        data["top_5_predictions"],
                    ]
                )

    print(f"Analysis complete. Results saved to {csv_filename}")


def visualize_species_plotly(csv_filename):
    """Generate and save an interactive Plotly bar chart of species counts."""
    df = pd.read_csv(csv_filename)
    df = df[df["refined_label"].notna() & (df["refined_label"] != "")]

    species_counts = df["refined_label"].value_counts().reset_index()
    species_counts.columns = ["Species", "Count"]
    species_counts = species_counts.sort_values(by="Count", ascending=True)

    dynamic_height = max(800, len(species_counts) * 20)
    fig = px.bar(
        species_counts,
        x="Count",
        y="Species",
        orientation="h",
        title='Species Distribution in "Birds" Album',
        color="Count",
        color_continuous_scale="Viridis",
        labels={"Count": "Number of Photos", "Species": "Bird Species"},
        height=dynamic_height,
    )
    fig.update_layout(
        title_font_size=24,
        yaxis={"dtick": 1},  # Show every species label
        margin=dict(l=200),  # Extra space for long bird names
        hovermode="y unified",
    )

    base_name = os.path.splitext(os.path.basename(csv_filename))[0]
    output_html = f"{base_name}.html"
    fig.write_html(output_html)
    print(f"Interactive visualization saved as {output_html}")
    fig.show()



def analyze_bird_data(csv_path, output_cm='confusion_matrix.png', output_pr='labeled_precision_recall_curve.png'):
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # -------------------------------------------------------------
    # 1. Number of images with No GPS data and number of birds = 1
    # -------------------------------------------------------------
    # Assumes 'location' is empty/NaN or a string indicating no data
    no_gps_mask = df['location'].isna() | (df['location'].astype(str).str.strip() == 'No GPS Data')
    birds_one_mask = df['bird_count'] == 1
    no_gps_single_bird = df[no_gps_mask & birds_one_mask].shape[0]
    
    print(f"1. Number of images with No GPS data and 1 bird: {no_gps_single_bird}")
    
    # -------------------------------------------------------------
    # 2. Number of images with no current_label
    # -------------------------------------------------------------
    no_current_label = df['current_label'].isna().sum()
    print(f"2. Number of images with no current_label: {no_current_label}")
    
    # -------------------------------------------------------------
    # 3. Clean 'Manual:' prefix and compute Confusion Matrix
    # -------------------------------------------------------------
    # Filter out records missing vital evaluation labels
    eval_df = df.dropna(subset=['current_label', 'top_label']).copy()
    
    # Clean the string prefixes
    eval_df['current_label'] = (
        eval_df['current_label']
        .astype(str)
        .str.replace(r'^Manual:\s*', '', regex=True)
        .str.strip()
    )
    eval_df['top_label'] = eval_df['top_label'].astype(str).str.strip()
    
    # Get a combined sorted list of unique species labels
    unique_labels = sorted(list(set(eval_df['current_label']).union(set(eval_df['top_label']))))
    
    # Compute the matrix
    cm = confusion_matrix(eval_df['current_label'], eval_df['top_label'], labels=unique_labels)
    
    # Plot the matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=unique_labels, yticklabels=unique_labels, cmap='Blues')
    plt.title('Bird Species Confusion Matrix (Cleaned Ground Truth vs Top Label)')
    plt.ylabel('Ground Truth (current_label)')
    plt.xlabel('Predicted (top_label)')
    plt.tight_layout()
    plt.savefig(output_cm, dpi=300, bbox_inches='tight')
    plt.show()
    
    y_true_strings = eval_df['current_label'].values
    y_pred_strings = eval_df['top_label'].values

    # Get the exact list of classes used in your confusion matrix
    classes = unique_labels  

    # 2. Map your single 'top_confidence' float column into a full (N_samples, N_classes) probability grid
    # This assigns the confidence score to the predicted class column, and 0 to all other columns
    y_scores_multiclass = np.zeros((len(eval_df), len(classes)))
    for i, (pred_label, conf) in enumerate(zip(y_pred_strings, eval_df['top_confidence'])):
        if pred_label in classes:
            class_idx = classes.index(pred_label)
            y_scores_multiclass[i, class_idx] = float(conf)

    # 3. Binarize the ground-truth labels into a matching (N_samples, N_classes) matrix
    y_true_multiclass = label_binarize(y_true_strings, classes=classes)

    # Handle edge case if your subset only contains 2 classes
    if y_true_multiclass.shape[1] == 1:
        y_true_multiclass = np.hstack((1 - y_true_multiclass, y_true_multiclass))

    # 4. Calculate the Micro-Averaged Precision-Recall Curve across all classes
    precision_micro, recall_micro, thresholds_micro = precision_recall_curve(
        y_true_multiclass.ravel(), 
        y_scores_multiclass.ravel()
    )

    # 5. Plot the corrected curve
    plt.figure(figsize=(10, 7))

    # 1. Plot the main blue line
    plt.plot(recall_micro, precision_micro, color='blue', lw=2, label='Micro-averaged PR Curve', zorder=1)

    # 2. Plot the color-coded dots
    sc = plt.scatter(recall_micro[:-1], precision_micro[:-1], c=thresholds_micro, 
                    cmap='viridis', s=35, zorder=2)
    cbar = plt.colorbar(sc)
    cbar.set_label('Confidence Score')

    # 3. Add explicit text numbers along the curve
    # We define target confidence levels we want to see written on the plot
    target_labels = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85]
    last_labeled_idx = -100  # Avoids crowding numbers together if points are too close

    for target in target_labels:
        # Find the index where the threshold is closest to our target step
        idx = np.argmin(np.abs(thresholds_micro - target))
        
        # Only label if it's a reasonable distance from the last labeled dot
        if abs(idx - last_labeled_idx) > 3 and idx < len(thresholds_micro):
            # Extract the coordinates for the text
            x = recall_micro[idx]
            y = precision_micro[idx]
            val = thresholds_micro[idx]
            
            # Draw the text label next to the dot
            plt.annotate(
                f"{val:.2f}", 
                xy=(x, y), 
                xytext=(7, 7),  # Offset text 7 points right and 7 points up from the dot
                textcoords='offset points', 
                fontsize=9, 
                fontweight='bold',
                color='black',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8) # Clean badge background
            )
            last_labeled_idx = idx

    # 4. Standard chart formatting
    plt.xlabel('Recall (Fraction of Birds Caught)')
    plt.ylabel('Precision (Accuracy / Freedom from False Positives)')
    plt.title('Multi-Class Precision-Recall Curve (With Direct Value Badges)')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])

    plt.tight_layout()
    plt.savefig(output_pr, dpi=300)
    plt.show()
    

def reconvert_to_predictions(summary_string):
    """Parse a top-5 summary string back into a list of (label, score) tuples."""
    if not isinstance(summary_string, str) or not summary_string.strip():
        return []

    predictions = []
    for entry in summary_string.split(" | "):
        match = re.search(r"\(([\d.]+)%\)$", entry.strip())
        if match:
            try:
                score_str = match.group(1)
                score = float(score_str) / 100.0
                label = entry.replace(f"({score_str}%)", "").strip()
                predictions.append((label, score))
            except ValueError:
                continue  # Skip malformed entries
    return predictions


def process_row(row):
    """Re-derive the refined label for a CSV row using stored predictions."""
    preds = row.get("top_5_predictions")
    if pd.isna(preds) or str(preds).strip() == "":
        return ""

    lat, lon = (float(x) for x in row["location"].split(", "))
    predictions = reconvert_to_predictions(str(preds))
    return get_refined_label(predictions, lat, lon, row["date"].split(" ")[0])


class StandaloneInferenceModel:
    def __init__(self, checkpoint_path, device="cpu"):
        self.device = torch.device(device)

        print(f"📂 Loading standalone checkpoint from {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        # 1. Recover the exact species list from training
        self.class_names = checkpoint["class_names"]
        num_classes = len(self.class_names)
        print(f"📋 Checkpoint verified for {num_classes} native species classes.")

        # 2. Build a custom container matching train_standalone.py's structural namespace
        class ModelContainer(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                # Must match the exact variable names used in train_standalone.py
                self.encoder = torch.hub.load(
                    "facebookresearch/dinov2", "dinov2_vitb14"
                )
                self.classifier = nn.Linear(768, num_classes)

            def forward(self, x):
                return self.classifier(self.encoder(x))

        self.model = ModelContainer(num_classes=num_classes)

        # 3. Use strict=True to guarantee every weight vector maps flawlessly
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.to(self.device).eval()
        print("✅ Standalone classifier weights successfully bound and active!")

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    256, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    @torch.no_grad()
    def predict(self, image_input, top_k=5):
        if isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        else:
            img = Image.open(image_input).convert("RGB")

        tensor = self.transform(img).unsqueeze(0).to(self.device)
        logits = self.model(tensor)

        probs = torch.softmax(logits, dim=-1).squeeze(0)
        topk_vals, topk_idx = torch.topk(probs, k=top_k)

        vals_list = topk_vals.cpu().tolist()
        idxs_list = topk_idx.cpu().tolist()

        predictions = []
        for val, idx in zip(vals_list, idxs_list):
            predictions.append((self.class_names[idx].replace("_", " "), val))
        return predictions


if __name__ == "__main__":
    print(f"Loading DETR on {device}...")
    detector = pipeline(
        "object-detection",
        model="facebook/detr-resnet-50",
        device=device,
        token=HF_API,
    )
    iqa_metric = pyiqa.create_metric("niqe", device=torch.device("cpu"))

    classifier_cfg = REGION_CONFIG.get(TARGET_REGION, REGION_CONFIG["US"])["classifier"]
    is_standalone = classifier_cfg.get("is_standalone")
    if is_standalone:
        hf_checkpoint_path = hf_hub_download(
            repo_id=classifier_cfg["repo_id"],
            filename=classifier_cfg["filename"],
        )
        classifier = StandaloneInferenceModel(hf_checkpoint_path, device=device)
    else:
        classifier = InferenceModel.from_pretrained(
            repo_id=classifier_cfg["repo_id"],
            filename=classifier_cfg["filename"],
        )

    base_name = os.path.splitext(os.path.basename(OUTPUT_FILE))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv_filename = f"{base_name}_{timestamp}.csv"
    process_birds_album(detector, classifier, iqa_metric, output_csv_filename)
    visualize_species_plotly(output_csv_filename)
    analyze_bird_data(output_csv_filename, output_cm=f"confusion_matrix_{timestamp}.png", output_pr=f"precision_recall_curve_{timestamp}.png")
    #sync_keywords_from_csv(output_csv_filename)
