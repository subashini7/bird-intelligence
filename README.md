# Apple Photos Bird Intelligence (APBI)

Apple Photos Bird Intelligence is a specialized pipeline that bridges the gap between raw AI object detection and expert-level birding logic. It scans your Apple Photos library, detects, counts, and classifies birds, then applies bio-geographic overrides to ensure your metadata reflects biological reality — not just AI guesses.

## 🚀 The Core Concept
Standard AI models often lack the context to know that a specific bird shouldn't exist in a certain location. This module uses spatial metadata and taxonomic grouping to "clean" identification results before writing them back to your Photos library as searchable keywords.

## ✨ Key Features
- **Apple Photos Integration:** Automatically writes species names as searchable keywords into your Photos app.
- **Multi-Region Support:** Configure for US, Singapore, India, or UK — each with region-specific classifiers, confidence thresholds, and species corrections.
- **Scientific-to-Common Name Conversion:** Non-US regions map scientific model output to common names via `regional_birds.csv`.
- **Bio-Geographic Overrides:** Spatial logic corrects species based on GPS coordinates (e.g., Island Scrub-Jay vs California Scrub-Jay).
- **Taxonomic Grouping:** Combines confidence scores for difficult-to-distinguish groups (Hummingbirds, Gulls, Grebes, Cranes).
- **Quality Scoring:** Uses **NIQE** (Natural Image Quality Evaluator) to score sharpness of bird crops. Note: images with water or heavy foliage can skew scores; this metric is still being refined.

## 📊 Logic Engine: Handling AI Inconsistencies

APBI doesn't just take the top AI result. It applies a layered set of rules to handle "look-alike" species complexes.

### 1. Spatial Logic (US Only)
If the AI identifies a **Western Scrub-Jay** but GPS coordinates place it in the **Channel Islands**, the script automatically renames it to the endemic **Island Scrub-Jay**.

### 2. Confidence Summing — Complex Groups (US Only)
For species that are notoriously difficult for AI to split, APBI sums the top two confidence scores. If the sum exceeds 99%, it uses a broader, more accurate label:

| AI Top 2 Candidates | Summed Conf | Final Label |
| :--- | :--- | :--- |
| Western Grebe / Clark's Grebe | > 99% | **Western/Clark's Grebe** |
| Any two Hummingbirds | > 99% | **Hummingbird** |
| Any two Gulls | > 99% | **Gull** |
| Subspecies of the same species | > 99% | **Base species name** |

### 3. Safety Labeling (US Only)
- **Green Heron:** Labeled as "Green Heron or Young Black-crowned Night-Heron" to account for frequent juvenile misidentification.
- **Cranes:** Generalized to "Crane" to prevent false positives for the endangered Whooping Crane.

### 4. Known Misclassification Corrections

Each region maintains a corrections table to fix systematic model errors:

**US**
| Model Predicts | Corrected To |
| :--- | :--- |
| Tennessee Warbler | Orange-crowned Warbler |
| Pileated Woodpecker | White-headed Woodpecker |
| Snow Goose | Ross's Goose |
| Surfbird | Dunlin |
| Downy Woodpecker | Hairy Woodpecker |
| Wilson's Phalarope | Red-necked Phalarope |

**UK** (confidence threshold: > 30%)
| Model Predicts | Corrected To |
| :--- | :--- |
| Whooper Swan | Mute Swan |
| Marsh Sandpiper | Common Greenshank |

**India** (confidence threshold: > 31%)
| Model Predicts | Corrected To |
| :--- | :--- |
| Great White Pelican | Spot-billed Pelican |
| Ring-billed Gull | Common Gull |
| Tawny Eagle | Black Kite |
| Dusky Crag-Martin | Little Cormorant |
| Thick-billed Flowerpecker | Ashy Woodswallow |
| Blue-cheeked Bee-eater | Blue-tailed Bee-eater |

## 🛠️ Architecture
```mermaid
graph TD
    A[Apple Photos] --> B[osxphotos Library]
    B --> C[DETR: Object Detection]
    C --> D{Bird Count = 1?}
    D -- Yes --> E{Check TARGET_REGION}
    D -- No --> I[Log Count Only]
    E -- US --> F1[Binocular Model from HF]
    E -- Singapore --> F2[StandaloneInferenceModel<br/>probe_best.pth]
    E -- India --> F3[StandaloneInferenceModel<br/>fine_tune_best.pth]
    E -- UK --> F4[StandaloneInferenceModel<br/>fine_tune_best.pth]
    F1 --> G[Species ID]
    F2 --> G
    F3 --> G
    F4 --> G
    G --> H{Apply Region Logic}
    H -- US --> J1[Geo + Taxonomy Rules<br/>Conf > 99%]
    H -- Singapore --> J2[Scientific → Common Name<br/>Conf > 40%]
    H -- India --> J3[Scientific → Common Name<br/>Corrections · Conf > 31%]
    H -- UK --> J4[Scientific → Common Name<br/>Corrections · Conf > 30%]
    J1 --> K[Set refined_label]
    J2 --> K
    J3 --> K
    J4 --> K
    K --> M[Write Keyword to Photos App]
```

## 💻 Setup & Installation

### Requirements
- **macOS** (required for Apple Photos access)
- **Python 3.11+**
- **Hugging Face API token** (`HF_TOKEN` in your `.env` file)

### Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file and add your Hugging Face token:
   ```
   HF_TOKEN=your_token_here
   ```

## 📝 Usage
Run the main script to process your "Birds" album:
```bash
python main.py
```
> **Note:** Ensure Photos is closed during database write operations to avoid conflicts.

## ⚙️ Configuration

Edit `TARGET_REGION` and `REGION_CONFIG` at the top of `main.py`:

```python
TARGET_REGION = "India"  # Options: "US", "Singapore", "India", "UK"

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
```

### Region Behaviour Summary
| Region | Model Source | Confidence Threshold | Name Conversion | Extra Logic |
| :--- | :--- | :--- | :--- | :--- |
| US | Binocular (HF) | 99% | — | Geo overrides, taxonomy grouping |
| Singapore | Standalone (HF) | 40% | Scientific → Common | — |
| India | Standalone (HF) | 31% | Scientific → Common | Species corrections |
| UK | Standalone (HF) | 30% | Scientific → Common | Species corrections |

## 🌍 Non-US Bird Data with iNaturalist
For regions outside the US, use `inaturalist.py` to download training data.
- Fetches bird observations from iNaturalist for a given region or place ID.
- Downloads and crops approximately 30 images per species.
- Saves images into folders named `processed_<region>_birds`.

```bash
python inaturalist.py
```

## 🧪 Training: Linear Probe and Fine-Tune DINOv2
After collecting regional images, use `dinov2_probe_fine_tune.py` to adapt the model:
- **Linear probe** — trains a classification head on frozen DINOv2 features.
- **Fine-tune** — unfreezes the encoder for deeper adaptation to local species.

Upload the resulting checkpoint to Hugging Face for use in `REGION_CONFIG`.

```bash
# Step 1: linear probe
python dinov2_probe_fine_tune.py --epochs 30 --lr 2e-4 --freeze_encoder --experiment_name probe

# Step 2: fine-tune from probe checkpoint
python dinov2_probe_fine_tune.py --epochs 30 --lr 1e-5 --resume <path_to_probe_best.pth> --experiment_name finetune
```

## 📈 Model Performance (US Region)

### Precision-Recall Curve
![Precision-Recall Curve](assets/US_precision_recall_curve_20260529_104121.png)

The curve shows model performance across the 0.99–1.00 confidence band used by the US pipeline.
At the operating threshold (conf ≥ 0.99): **P=0.95, R=0.89** across 176 species.

### Confusion Matrix
The interactive confusion matrix (species ordered by taxonomic sequence) can be viewed here:
👉 [Open Interactive Confusion Matrix](assets/US_confusion_matrix_20260529_104121.html)

> **Note:** The HTML confusion matrix must be viewed locally or via GitHub Pages — 
> GitHub's README renderer does not display raw HTML files inline.

## ⚖️ License
This project is licensed under the MIT License. Models used: [Facebook DETR](https://huggingface.co/facebook/detr-resnet-50) and [Binocular Bird Classifier](https://huggingface.co/jiujiuche/binocular).
