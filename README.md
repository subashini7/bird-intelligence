# Apple Photos Bird Intelligence - APBI

Apple Photos Bird Intelligence is a specialized pipeline that bridges the gap between raw AI object detection and expert-level birding logic. It scans your Apple Photos library, detects, counts and classifies birds, and applies bio-geographic overrides to ensure your metadata reflects biological reality, not just AI guesses.

## 🚀 The Core Concept
Standard AI models often lack the context to know that a specific bird shouldn't exist in a certain location. This module uses spatial metadata and taxonomic grouping to "clean" identification results before writing them back to your Photos library as searchable keywords.

## ✨ Key Features
- **Apple Photos Integration:** Automatically writes species names as searchable keywords (Keywords/Tags) into your Photos app.
- **Bio-Geographic Overrides:** Spatial logic to correct species based on location (e.g., Island Scrub-Jay corrections).
- **Taxonomic Grouping:** Combines confidence scores for difficult-to-distinguish groups (Hummingbirds, Gulls, Grebes).
- **Quality Scoring:** Uses **NIQE** (Natural Image Quality Evaluator) to determine the sharpness of bird crops. Note that this still needs improvement. The image gets a good score when there are too many branches or a lower score if the bird is floating in water.
- **eBird Verification:** Cross-references detections with real-time local sightings via the eBird API. TODO

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
    D -- Yes --> E[Binocular: Species ID]
    E --> F[Logic Engine: Geo + Taxonomy]
    F --> G[eBird API Verification]
    G --> H[Write Keyword to Photos App]
    D -- No --> I[Log Count Only]
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
3. Set your API keys in the script:
   - `HF_API`: Your Hugging Face token.
   - `EBIRD_API_KEY`: Your eBird developer key.

## 📝 Usage
Run the main script to process your "Birds" album:
```bash
python main.py
```
*Note: Ensure the Photos App is closed during database write operations.*

## ⚖️ License
This project is licensed under the MIT License. Models used: [Facebook DETR](https://huggingface.co) and [Binocular Bird Classifier](https://huggingface.co).
