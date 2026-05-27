# Apple Photos Bird Intelligence - APBI

Apple Photos Bird Intelligence is a specialized pipeline that bridges the gap between raw AI object detection and expert-level birding logic. It scans your Apple Photos library, detects, counts and classifies birds, and applies bio-geographic overrides to ensure your metadata reflects biological reality, not just AI guesses.

## 🚀 The Core Concept
Standard AI models often lack the context to know that a specific bird shouldn't exist in a certain location. This module uses spatial metadata and taxonomic grouping to "clean" identification results before writing them back to your Photos library as searchable keywords.

## ✨ Key Features
- **Apple Photos Integration:** Automatically writes species names as searchable keywords (Keywords/Tags) into your Photos app.
- **Multi-Region Support:** Configure for different regions (US, Singapore) with region-specific classifiers and scientific-to-common name conversion.
- **Bio-Geographic Overrides:** Spatial logic to correct species based on location (e.g., Island Scrub-Jay corrections).
- **Taxonomic Grouping:** Combines confidence scores for difficult-to-distinguish groups (Hummingbirds, Gulls, Grebes).
- **Quality Scoring:** Uses **NIQE** (Natural Image Quality Evaluator) to determine the sharpness of bird crops. Note that this still needs improvement. The image gets a good score when there are too many branches or a lower score if the bird is floating in water.

## 📊 Logic Engine: Handling AI Inconsistencies
APBI doesn't just take the top AI result. It applies specific rules to handle "look-alike" complexes:

### 1. Spatial Logic (The Island Rule)
If the AI identifies a **Western Scrub-Jay** but the GPS coordinates place it in the **Channel Islands**, the script automatically renames it to the endemic **Island Scrub-Jay**.

### 2. Confidence Summing (Complex Groups)
For species that are notoriously difficult for AI to split, APBI sums the top two results. If the sum is > 99%, it uses a broader, more accurate label:


| AI Top 2 Candidates | Summed Conf | Final Label |
| :--- | :--- | :--- |
| Western Grebe / Clark's Grebe | > 99% | **Western/Clark Grebe** |
| Any two Hummingbirds | > 99% | **Hummingbird** |
| Gulls / Terns / Loons | > 99% | **[Group Name]** |
| Sandhill / Whooping Crane | > 99% | **Crane** |

### 3. Safety Labeling
- **Green Heron:** Labeled as "Green Heron or Young Black-crowned Night-Heron" to account for frequent juvenile misidentification.
- **Cranes:** Detections are generalized to "Crane" to prevent false positives for the endangered Whooping Crane.

## 🛠️ Architecture
```mermaid
graph TD
    A[Apple Photos] --> B[osxphotos Library]
    B --> C[DETR: Object Detection]
    C --> D{Bird Count = 1?}
    D -- Yes --> E{Check TARGET_REGION}
    D -- No --> I[Log Count Only]
    E -- US --> F1[Load Binocular from HF]
    E -- Singapore --> F2[Load Local Model<br/>singapore_probe_best.pth]
    F1 --> G1[Species ID]
    F2 --> G1
    G1 --> H{Region Config}
    H -- US --> J1[Apply Geo + Taxonomy<br/>Logic Engine]
    H -- Singapore --> J2[Convert Scientific<br/>to Common Name]
    J1 --> K1{Confidence > 99%?}
    J2 --> K2{Confidence > 65%?}
    K1 -- Yes --> L1[Set refined_label]
    K2 -- Yes --> L2[Set refined_label]
    K1 -- No --> L3[No Label]
    K2 -- No --> L3
    L1 --> M[Write Keyword to<br/>Photos App]
    L2 --> M
    L3 --> M
```

## 💻 Setup & Installation

### Requirements
- **macOS** (For Apple Photos access)
- **Python 3.11+**
- **eBird API Key**

### Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your API key in the script:
   - `HF_API`: Your Hugging Face token.

## 📝 Usage
Run the main script to process your "Birds" album:
```bash
python main.py
```
*Note: Ensure the Photos App is closed during database write operations.*

## 🌍 Non-US Bird Data with iNaturalist
If your birds are outside the US, use `inaturalist.py` to download species data for a country or region.
- `inaturalist.py` can fetch bird observations from iNaturalist for a given region.
- It can save around 30 bird images per species by downloading and cropping candidate photos.

Example:
```bash
python inaturalist.py
```
This script currently supports region-specific place IDs and saves images into folders like `processed_<region>_birds`.

## 🧪 Linear Probe and Fine-Tune DINOv2
After collecting non-US bird images, use `dinov2_probe_fine_tune.py` to:
- linear probe the DINOv2 model
- fine-tune DINOv2 on your region-specific bird data

This makes the model more adapted to your local species and image distribution.

Example:
```bash
python dinov2_probe_fine_tune.py --epochs 30 --lr 2e-4 --freeze_encoder --experiment_name probe
!python dinov2_probe_fine_tune.py --epochs 30 --lr 1e-5 --resume /kaggle/working/checkpoints/probe_best.pth --experiment_name finetune
```

## 🔧 About `main.py`
`main.py` now supports region-specific classifiers and scientific-to-common name conversion. The script can be configured to use different bird classifiers for different regions (e.g., US or Singapore).

### Configuration
Edit the `TARGET_REGION` and `REGION_CONFIG` at the top of `main.py`:

```python
TARGET_REGION = "Singapore"  # Set to "US" or "Singapore"

REGION_CONFIG = {
    "US": {
        "country_codes": {"US"},
        "classifier": {
            "repo_id": "jiujiuche/binocular",
            "filename": "artifacts/dinov2_vitb14_nabirds.pth",
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
}
```

### Region-Specific Features
- **US Region:** Uses the default Binocular classifier with geographic overrides.
- **Singapore Region:** 
  - Loads a fine-tuned local model (`singapore_probe_best.pth`) to work with Singapore birds.
  - Converts scientific species names to common names using `regional_birds.csv`.
  - Uses a simplified refined label rule: if top-1 confidence > 65%, apply that label.

## ⚖️ License
This project is licensed under the MIT License. Models used: [Facebook DETR](https://huggingface.co) and [Binocular Bird Classifier](https://huggingface.co).