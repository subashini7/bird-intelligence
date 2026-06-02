import csv
import os
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
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.preprocessing import label_binarize
from torchvision import transforms
from transformers import pipeline

load_dotenv()
HF_API = os.getenv("HF_TOKEN")
BIRD_TAG_PREFIX = "Bird: "
MANUAL_PREFIX = "Manual: "
device = "mps" if torch.backends.mps.is_available() else "cpu"
OUTPUT_FILE = "classified_birds_report.csv"
ALBUM_NAME = "Birds"
DATE_CUTOFF = date(2024, 3, 28)
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
        "confidence_threshold_for_plot": 0.99,
        "pr_thresholds": [0.5, 0.990, 0.995, 1.000],
        "composite_labels": {
            "Gull",
            "Hummingbird",
            "Crane",
            "Western/Clark's Grebe",
            "Green Heron or Young Black-crowned Night-Heron",
        },
        "composite_constituent_species": {
            "Anna's Hummingbird",
            "Allen's Hummingbird",
            "Rufous Hummingbird",
            "Black-chinned Hummingbird",
            "Calliope Hummingbird",
            "Costa's Hummingbird",
        },
    },
    "Singapore": {
        "country_codes": {"SG"},
        "classifier": {
            "repo_id": "pshops/dinov2-singapore-birds",
            "filename": "probe_best.pth",
            "is_standalone": True,
        },
        "use_scientific_to_common": True,
        "confidence_threshold_for_plot": 0.40,
        "pr_thresholds": [0.40, 0.55, 0.70],
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
        "confidence_threshold_for_plot": 0.31,
        "pr_thresholds": [0.31, 0.50, 0.70],
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
        "confidence_threshold_for_plot": 0.31,
        "pr_thresholds": [0.30, 0.50, 0.70],
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


def load_date_corrections(csv_path=None):
    """Load per-date species corrections from us_date_corrections.csv."""
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(__file__), "us_rare_bird_date_corrections.csv")
    if not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    return {
        (str(row["date"]).strip(), str(row["from_label"]).strip()): str(row["to_label"]).strip()
        for _, row in df.iterrows()
    }


DATE_CORRECTIONS = load_date_corrections()


def get_common_name(scientific_name: str) -> str:
    base_name = scientific_name.split(" (")[0].strip()
    return SCIENTIFIC_TO_COMMON.get(
        base_name, SCIENTIFIC_TO_COMMON.get(scientific_name, scientific_name)
    )


def convert_predictions_to_common(predictions):
    return [(get_common_name(label), score) for label, score in predictions]


def normalize_label(label: str) -> str:
    """Normalize hyphens and spelling variants for cross-region species name comparison."""
    return label.replace("-", " ").replace("Gray", "Grey").strip()


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
        manual_tags = [k for k in current_keywords if k.startswith(MANUAL_PREFIX)]
        cleaned_keywords = [k for k in current_keywords if not is_bird_tag(k) and not k.startswith(MANUAL_PREFIX)]

        if fname not in label_map:
            not_found += 1
            continue

        new_label = label_map[fname]

        # Split "Species1 / Species2" into individual tags; deduplicate same species.
        new_tags = list(dict.fromkeys(
            make_tag(s.strip()) for s in new_label.split(" / ") if s.strip()
        ))
        new_species = {s.strip() for s in new_label.split(" / ") if s.strip()}

        if manual_tags:
            manual_species = {t.removeprefix(MANUAL_PREFIX).strip() for t in manual_tags}
            if {normalize_label(s).lower() for s in manual_species} != {normalize_label(s).lower() for s in new_species}:
                # At least one manual tag disagrees — leave everything untouched
                continue
            # Every manual tag matches — replace all "Manual: X" with Bird: tag(s)
            if new_tags:
                print(f"  {fname}: Manual tags {sorted(manual_species)} → '{new_label}'")
                photo.keywords = cleaned_keywords + new_tags
                updated += 1
            else:
                photo.keywords = cleaned_keywords
                cleared += 1
        else:
            # No manual tag — compare species sets to avoid false positives from order differences
            existing_bird_tags = [k for k in current_keywords if is_bird_tag(k)]
            existing_species = {t.removeprefix("Bird: ").strip() for t in existing_bird_tags}

            if {normalize_label(s) for s in existing_species} == {normalize_label(s) for s in new_species}:
                continue

            if new_tags:
                print(f"  {fname}: '{' / '.join(sorted(existing_species))}' → '{new_label}'")
                photo.keywords = cleaned_keywords + new_tags
                updated += 1
            else:
                photo.keywords = cleaned_keywords
                cleared += 1

    print(f"Done. Updated: {updated} | Cleared: {cleared} | Not in CSV: {not_found}")


def get_refined_us_label(predictions, lat, lon, date):
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
        "Whimbrel": "Hudsonian Whimbrel",
    }
    if base_refined_label in corrections:
        refined_label = corrections[base_refined_label]
    elif (date, base_refined_label) in DATE_CORRECTIONS:
        refined_label = DATE_CORRECTIONS[(date, base_refined_label)]
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


def detect_classify(photo, is_in_region, detector, classifier, iqa_metric):
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
    bird_kws = sorted(k.replace(BIRD_TAG_PREFIX, "") for k in photo.keywords if is_bird_tag(k))
    if bird_kws:
        current_label = " / ".join(bird_kws)
    else:
        # Bird: tags are written by sync_keywords_from_csv which runs after this scan.
        # Use Manual: tags as ground truth when Bird: tags haven't been synced yet.
        manual_kws = sorted(k.removeprefix(MANUAL_PREFIX).strip() for k in photo.keywords if k.startswith(MANUAL_PREFIX))
        current_label = " / ".join(manual_kws) if manual_kws else ""

    try:
        if (
            not photo.path
            or not os.path.exists(str(photo.path))
            or not is_in_region
        ):
            return None
        photo_path = photo.path_edited if photo.path_edited else photo.path
        photo_path = str(photo_path)
        image = Image.open(photo_path).convert("RGB")
        image.load()  # Force load to catch "premature end of data"

        results = detector(photo_path)
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
            bird_labels = []  # (refined_label, top_label, top_confidence, top_5_summary)
            loc_data = photo.location

            # DETR's pipeline loads via PIL (EXIF-aware); cv2.imread ignores EXIF
            # rotation.  Clamp and crop in PIL space so coordinates always match.
            pil_w, pil_h = image.size
            image_area = pil_w * pil_h

            for b in birds:
                box = b["box"]
                x1 = max(0, int(box["xmin"]))
                y1 = max(0, int(box["ymin"]))
                x2 = min(pil_w, int(box["xmax"]))
                y2 = min(pil_h, int(box["ymax"]))
                crop_w, crop_h = x2 - x1, y2 - y1

                # Reject crops that are too small in absolute terms or relative to
                # the image — filters foliage blobs, reflections, and distant specks.
                if crop_w < 48 or crop_h < 48 or (crop_w * crop_h) < 0.001 * image_area:
                    continue

                crop_pil = image.crop((x1, y1, x2, y2))
                crop_cv = cv2.cvtColor(np.array(crop_pil), cv2.COLOR_RGB2BGR)
                try:
                    scores.append(get_bird_quality_score(crop_cv, iqa_metric))
                except Exception:
                    scores.append(0.0)

                if len(birds) <= 2 and is_in_region:
                    predictions = classifier.predict(crop_pil, top_k=5)
                    if REGION_CONFIG.get(TARGET_REGION, {}).get(
                        "use_scientific_to_common", False
                    ):
                        predictions = convert_predictions_to_common(predictions)
                    bird_refined = ""
                    bird_top_label, bird_top_conf = predictions[0]
                    if TARGET_REGION == "Singapore":
                        if bird_top_conf > 0.40:
                            bird_refined = bird_top_label
                    elif TARGET_REGION == "UK":
                        if bird_top_conf > 0.31:
                            bird_refined = bird_top_label
                        corrections = {
                            "Whooper Swan": "Mute Swan",
                            "Ring-billed Gull": "Common Gull",
                        }
                        if bird_refined in corrections:
                            bird_refined = corrections[bird_refined]
                    elif TARGET_REGION == "India":
                        if bird_top_conf > 0.31:
                            bird_refined = bird_top_label
                        # There are confusions between Whiskered and Gull-billed Tern;
                        # Little and Indian Cormorant; Little, Medium and Great Egret;
                        # Inidan Robin => Pied Bushcat; Pied Bushcat => Black Drongo
                        # The above labels need to be reviewed manually
                        corrections = {
                            "Great White Pelican": "Spot-billed Pelican",
                            "Marsh Sandpiper": "Common Greenshank",
                            "Tawny Eagle": "Black Kite",
                            "Dusky Crag-Martin": "Little Cormorant",
                            "Thick-billed Flowerpecker": "Ashy Woodswallow",
                            "Blue-cheeked Bee-eater": "Blue-tailed Bee-eater",
                            "Western Yellow Wagtail": "Eastern Yellow Wagtail",
                            "Lesser Flamingo": "Greater Flamingo",
                        }
                        if bird_refined in corrections:
                            bird_refined = corrections[bird_refined]
                    elif TARGET_REGION == "US":
                        bird_refined = get_refined_us_label(
                            predictions,
                            loc_data[0],
                            loc_data[1],
                            photo.date.strftime("%Y-%m-%d"),
                        )

                    bird_top_5 = " | ".join(f"{label} ({score:.2%})" for label, score in predictions)
                    bird_labels.append((bird_refined, bird_top_label, bird_top_conf, bird_top_5))

                    print(f"\n--- Predictions for {display_name} (bird {len(bird_labels)}) ---")
                    for label, score in predictions:
                        print(f"Species: {label:30} | Confidence: {score:.2%}")
                    cv2.putText(
                        img_cv,
                        bird_refined,
                        (x1, y1 - 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (255, 255, 0),
                        3,
                    )

            # Merge per-bird results into photo-level fields
            if len(bird_labels) == 1:
                refined_label, top_label, top_confidence, top_5_summary = bird_labels[0]
            elif len(bird_labels) == 2:
                r1, t1, c1, s1 = bird_labels[0]
                r2, t2, c2, s2 = bird_labels[1]
                (r_hi, t_hi, c_hi, s_hi), (r_lo, _, _, _) = (
                    (bird_labels[0], bird_labels[1]) if c1 >= c2
                    else (bird_labels[1], bird_labels[0])
                )
                top_label, top_confidence, top_5_summary = t_hi, c_hi, s_hi
                if not r_hi and not r_lo:
                    refined_label = ""
                elif not r_lo or r_hi == r_lo:
                    refined_label = r_hi
                elif not r_hi:
                    refined_label = r_lo
                else:
                    refined_label = f"{r_hi} / {r_lo}"

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


def assign_bird_group(species_name):
    name = str(species_name).lower()

    # 1. Waders — egrets, herons, ibis, spoonbills, bitterns
    if any(x in name for x in ["egret", "heron", "ibis", "spoonbill", "bittern"]):
        return "1_Waders_Egrets_Herons"
    # 2. Pelicans and large seabirds
    elif any(x in name for x in ["pelican", "gannet", "booby", "frigatebird"]):
        return "2_Pelicans_Seabirds"
    # 31. Water divers — cormorants, grebes, loons, alcids
    elif any(x in name for x in ["cormorant", "shag", "grebe", "loon", "murre",
                                  "guillemot", "auklet", "puffin", "murrelet", "razorbill"]):
        return "31_WaterDivers"
    # 32. Gulls
    elif any(x in name for x in ["gull", "kittiwake"]):
        return "32_Gulls"
    # 33. Terns
    elif "tern" in name:
        return "33_Terns"
    # 34. Waterfowl — ducks, geese, swans; includes diving ducks missing from original
    elif any(x in name for x in ["duck", "wigeon", "teal", "shelduck", "scoter", "merganser",
                                  "goose", "mallard", "goldeneye", "canvasback", "bufflehead",
                                  "scaup", "redhead", "eider", "pintail", "shoveler"]):
        return "34_Ducks"
    # 35. Rails and coots
    elif any(x in name for x in ["rail", "sora", "gallinule", "coot"]):
        return "35_Rails"
    # 36. Shorebirds — stilts, avocets, plovers, sandpipers, phalaropes, etc.
    elif any(x in name for x in ["stilt", "avocet", "plover", "sandpiper", "dowitcher",
                                  "yellowlegs", "willet", "turnstone", "dunlin", "oystercatcher",
                                  "godwit", "curlew", "whimbrel", "sanderling", "phalarope",
                                  "snipe", "woodcock", "surfbird", "tattler", "knot"]):
        return "36_Shorebirds"
    # 4. Woodpeckers, flickers, sapsuckers
    elif any(x in name for x in ["woodpecker", "flicker", "sapsucker"]):
        return "4_Woodpeckers"
    # 5. Hummingbirds
    elif "hummingbird" in name:
        return "5_Hummingbirds"
    # 8. Raptors — checked before swallows to catch Swallow-tailed Kite
    elif any(x in name for x in ["eagle", "hawk", "falcon", "kite", "vulture",
                                  "shikra", "condor", "osprey", "harrier"]):
        return "8_Raptors"
    # 61. Corvids
    elif any(x in name for x in ["jay", "magpie", "crow", "raven", "nutcracker", "chough"]):
        return "61_Crows"
    # 62. Blackbirds, grackles, cowbirds, orioles (US + India)
    elif any(x in name for x in ["grackle", "cowbird", "blackbird", "oriole", "meadowlark",
                                  "koel", "drongo", "bushchat", "bulbul", "shrike", "minivet", "sunbird"]):
        return "62_Blackbirds"
    # 63. Thrushes, robins, flycatchers, warblers, swallows, martins
    elif any(x in name for x in ["robin", "thrush", "bluebird", "flycatcher", "warbler",
                                  "swallow", "martin"]):
        return "63_Thrushes_Warblers"
    # 7. Small passerines — sparrows, finches, buntings, juncos, towhees (US + India)
    elif any(x in name for x in ["sparrow", "finch", "bunting", "junco", "towhee",
                                  "bee-eater", "leafbird"]):
        return "7_Small_Birds"
    # 9. Parakeets and parrots
    elif any(x in name for x in ["parakeet", "parrot"]):
        return "9_Parakeets"
    else:
        return "99_Other_Species"

def analyze_bird_data(csv_path, output_cm="confusion_matrix.png", output_pr="labeled_precision_recall_curve.png",
                      output_f1="f1_by_species.html", output_conf="confidence_histogram.png",
                      output_cov="coverage_precision.png"):
    """Compute and save a confusion matrix and precision-recall curve for classified birds."""
    df = pd.read_csv(csv_path)

    # Number of images with no GPS data and exactly one bird detected
    no_gps_mask = df["location"].isna() | (df["location"].astype(str).str.strip() == "No GPS Data")
    birds_one_mask = df["bird_count"] == 1
    no_gps_single_bird = df[no_gps_mask & birds_one_mask].shape[0]
    print(f"1. Number of images with No GPS data and 1 bird: {no_gps_single_bird}")

    # Number of single-bird images with no current_label
    no_current_label = df[birds_one_mask]["current_label"].isna().sum()
    print(f"2. Number of images with no current_label: {no_current_label}")

    # Require ground truth (current_label) but allow empty refined_label — species
    # the model never predicts above threshold would otherwise be invisible in F1.
    eval_df = df.dropna(subset=["current_label"]).copy()
    eval_df["current_label"] = (
        eval_df["current_label"]
        .astype(str)
        .str.replace(r"^Manual:\s*", "", regex=True)
        .str.split(" (", regex=False)
        .str[0]
        .str.strip()
    )
    eval_df = eval_df[eval_df["current_label"] != ""].copy()
    # Empty refined_label (below-threshold photos) becomes "" — treated as no prediction
    eval_df["refined_label"] = (
        eval_df["refined_label"]
        .fillna("")
        .astype(str)
        .str.split(" (", regex=False)
        .str[0]
        .str.strip()
    )
    # Normalize hyphens so "Black-crowned Night-Heron" == "Black-crowned Night Heron"
    eval_df["current_label"] = eval_df["current_label"].apply(normalize_label)
    eval_df["refined_label"] = eval_df["refined_label"].apply(
        lambda x: normalize_label(x) if x else x
    )
    eval_df["top_confidence"] = pd.to_numeric(eval_df["top_confidence"], errors="coerce").fillna(0)

    # Expand 2-bird rows into individual species rows for evaluation.
    # Matched species (appear in both current and refined) are paired as correct.
    # Species only in current_label become (species, "") — missed detections.
    # Species only in refined_label become ("", species) — spurious detections.
    expanded_rows = []
    for _, row in eval_df.iterrows():
        cp = sorted(s.strip() for s in str(row["current_label"]).split(" / ") if s.strip())
        rp = sorted(s.strip() for s in str(row["refined_label"]).split(" / ") if s.strip())
        if len(cp) > 1 or len(rp) > 1:
            cp_remaining = list(cp)
            rp_remaining = list(rp)
            # First pass: match species present in both lists
            for species in list(cp_remaining):
                if species in rp_remaining:
                    nr = row.copy()
                    nr["current_label"] = species
                    nr["refined_label"] = species
                    expanded_rows.append(nr)
                    cp_remaining.remove(species)
                    rp_remaining.remove(species)
            # Remaining current species had no prediction
            for species in cp_remaining:
                nr = row.copy()
                nr["current_label"] = species
                nr["refined_label"] = rp_remaining.pop(0) if rp_remaining else ""
                expanded_rows.append(nr)
            # Any leftover predictions have no ground truth
            for pred in rp_remaining:
                nr = row.copy()
                nr["current_label"] = ""
                nr["refined_label"] = pred
                expanded_rows.append(nr)
        else:
            expanded_rows.append(row)
    eval_df = pd.DataFrame(expanded_rows, columns=eval_df.columns).reset_index(drop=True)
    eval_df = eval_df[eval_df["current_label"] != ""].copy()

    # Exclude composite/grouped labels from all evaluation — they represent
    # intentional ambiguity, not model errors, and distort F1 and confusion matrix.
    composite_labels = {normalize_label(l) for l in REGION_CONFIG[TARGET_REGION].get("composite_labels", set())}
    if composite_labels:
        excluded = eval_df["refined_label"].isin(composite_labels) | eval_df["current_label"].isin(composite_labels)
        print(f"Excluded {excluded.sum()} composite-label rows from evaluation.")
        eval_df = eval_df[~excluded].copy()

    # Exclude constituent species whose photos are predominantly routed to a composite
    # label — their remaining rows (below threshold) would give a misleading F1=0.
    composite_constituents = {normalize_label(l) for l in REGION_CONFIG[TARGET_REGION].get("composite_constituent_species", set())}
    if composite_constituents:
        excluded_c = eval_df["current_label"].isin(composite_constituents)
        print(f"Excluded {excluded_c.sum()} constituent-species rows from evaluation.")
        eval_df = eval_df[~excluded_c].copy()

    filtered_df = eval_df[
        (eval_df["top_confidence"] >= REGION_CONFIG[TARGET_REGION]["confidence_threshold_for_plot"]) &
        (eval_df["refined_label"] != "")
    ]
    all_labels = sorted(list(set(filtered_df["current_label"]).union(set(filtered_df["refined_label"]))))
    raw_cm = confusion_matrix(filtered_df["current_label"], filtered_df["refined_label"], labels=all_labels)
    if raw_cm.sum() > 0:
        cm_df_raw = pd.DataFrame(raw_cm, index=all_labels, columns=all_labels)

        # Identify species with at least one misclassification
        cm_raw_copy = np.copy(cm_df_raw.values)
        np.fill_diagonal(cm_raw_copy, 0)
        cm_errors_only = pd.DataFrame(cm_raw_copy, index=all_labels, columns=all_labels)
        wrong_label_mask = (cm_errors_only.sum(axis=0) > 0) | (cm_errors_only.sum(axis=1) > 0)
        active_species = cm_df_raw.loc[wrong_label_mask, wrong_label_mask].index.tolist()

        # Sort active species by taxonomy group
        sorting_df = pd.DataFrame({"species": active_species})
        sorting_df["group"] = sorting_df["species"].apply(assign_bird_group)
        sorting_df = sorting_df.sort_values(by=["group", "species"])
        grouped_sorted_labels = sorting_df["species"].tolist()
        if not grouped_sorted_labels:
            print("Skipping confusion matrix: no misclassified species to plot.")
        else:
            # Build and save the Plotly interactive confusion matrix
            final_cm = confusion_matrix(filtered_df["current_label"], filtered_df["refined_label"], labels=grouped_sorted_labels)
            fig, ax = plt.subplots(figsize=(11, 10))

            im = ax.imshow(final_cm, cmap="viridis")
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Count", fontsize=10)
            ax.set_title("Confusion Matrix", fontsize=12, pad=15)
            ax.set_xlabel("Predicted Species", fontsize=10)
            ax.set_ylabel("Ground Truth", fontsize=10)
            ax.set_xticks(range(len(grouped_sorted_labels)))
            ax.set_yticks(range(len(grouped_sorted_labels)))
            ax.set_xticklabels(grouped_sorted_labels, rotation=-90, fontsize=10)
            ax.set_yticklabels(grouped_sorted_labels, fontsize=10)
            plt.tight_layout()
            plt.savefig(output_cm, dpi=300, bbox_inches="tight")
            plt.close()

    y_true_strings = eval_df["current_label"].values
    y_pred_strings = eval_df["refined_label"].values

    # Build the full label set used in the confusion matrix
    unique_labels = sorted(list(set(eval_df["current_label"]).union(set(eval_df["refined_label"]))))

    # Map top_confidence into an (N_samples, N_classes) probability grid:
    # the predicted class column gets the confidence score; all others get 0.
    y_scores_multiclass = np.zeros((len(eval_df), len(unique_labels)))
    for i, (pred_label, conf) in enumerate(zip(y_pred_strings, eval_df["top_confidence"])):
        if pred_label in unique_labels:
            class_idx = unique_labels.index(pred_label)
            y_scores_multiclass[i, class_idx] = float(conf)
 
    # Binarize the ground-truth labels into a matching (N_samples, N_classes) matrix
    y_true_multiclass = label_binarize(y_true_strings, classes=unique_labels)
 
    # Handle edge case when the subset contains only 2 classes
    if y_true_multiclass.shape[1] == 1:
        y_true_multiclass = np.hstack((1 - y_true_multiclass, y_true_multiclass))
 
    # Compute the micro-averaged precision-recall curve across all classes
    precision_micro, recall_micro, thresholds_micro = precision_recall_curve(
        y_true_multiclass.ravel(),
        y_scores_multiclass.ravel()
    )
 
    fig_pr, ax = plt.subplots(figsize=(10, 7))
 
    sc = ax.scatter(recall_micro[:-1], precision_micro[:-1], c=thresholds_micro,
                    cmap="viridis", s=60, zorder=3)
    ax.plot(recall_micro, precision_micro, color="royalblue", lw=2,
            label="Micro-averaged PR Curve", zorder=2)
    cbar = fig_pr.colorbar(sc, ax=ax)
    cbar.set_label("Confidence Score")
 
    # Annotate key operating points with confidence, precision, and recall
    n_points = len(thresholds_micro)
    target_thresholds = REGION_CONFIG[TARGET_REGION]["pr_thresholds"]
    offsets = [(10, -18), (10, 6), (-55, 10)]
    for t, offset in zip(target_thresholds, offsets):
        idx = np.argmin(np.abs(thresholds_micro - t))
        if idx < n_points:
            ax.annotate(
                f"conf={thresholds_micro[idx]:.3f}\nP={precision_micro[idx]:.2f} R={recall_micro[idx]:.2f}",
                xy=(recall_micro[idx], precision_micro[idx]),
                xytext=offset, textcoords="offset points",
                fontsize=8.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85),
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
            )
 
    # Reference line for exact-match accuracy
    accuracy = (eval_df["current_label"] == eval_df["top_label"]).mean()
    n_labeled = len(eval_df)
    n_species = len(unique_labels)
    ax.axhline(accuracy, color="tomato", linestyle="--", lw=1.2,
               label=f"Exact-match accuracy: {accuracy:.1%}")
 
    ax.set_xlabel("Recall (Fraction of Birds Caught)", fontsize=12)
    ax.set_ylabel("Precision (Accuracy / Freedom from False Positives)", fontsize=12)
    ax.set_title(
        f"Multi-Class Precision-Recall Curve\n"
        f"(parentheticals stripped — {n_labeled} labelled photos, {n_species} species)",
        fontsize=12,
    )
    pr_y_min = max(0.0, float(np.nanmin(precision_micro)) - 0.05)
    ax.set_xlim([-0.02, 1.05])
    ax.set_ylim([pr_y_min, 1.02])
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_pr, dpi=180)
    plt.close()

    # mAP
    try:
        mAP = average_precision_score(y_true_multiclass, y_scores_multiclass, average="macro")
        print(f"mAP (macro-averaged): {mAP:.3f}")
    except Exception:
        pass

    # Per-species F1 chart
    report = classification_report(
        eval_df["current_label"], eval_df["refined_label"],
        output_dict=True, zero_division=0,
    )
    n_unlabeled = (eval_df["refined_label"] == "").sum()
    if n_unlabeled:
        print(f"{n_unlabeled} photos had a ground-truth label but no prediction (below confidence threshold).")
    f1_rows = [
        {"Species": k, "F1": v["f1-score"], "Precision": v["precision"],
         "Recall": v["recall"], "Support": int(v["support"])}
        for k, v in report.items()
        if k not in {"accuracy", "macro avg", "weighted avg", "micro avg", ""}
    ]
    f1_df = pd.DataFrame(f1_rows).sort_values("F1")
    fig_f1 = px.bar(
        f1_df, x="F1", y="Species", orientation="h",
        title=f"Per-Species F1 Score — {TARGET_REGION}",
        color="F1", color_continuous_scale="RdYlGn",
        hover_data={"Precision": ":.2f", "Recall": ":.2f", "Support": True},
        range_x=[0, 1],
        height=max(600, len(f1_df) * 22),
    )
    fig_f1.update_layout(yaxis={"dtick": 1}, margin=dict(l=200))
    fig_f1.write_html(output_f1)
    print(f"Per-species F1 chart saved as {output_f1}")

    # Confidence histogram — correct vs incorrect predictions
    op_thresh = REGION_CONFIG[TARGET_REGION]["confidence_threshold_for_plot"]
    eval_df["correct"] = eval_df["current_label"] == eval_df["refined_label"]
    correct_conf = eval_df[eval_df["correct"]]["top_confidence"]
    wrong_conf = eval_df[~eval_df["correct"]]["top_confidence"]
    fig_hist, ax_h = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0, 1, 30)
    ax_h.hist(correct_conf, bins=bins, alpha=0.6, color="seagreen",
              label=f"Correct ({len(correct_conf)})")
    ax_h.hist(wrong_conf, bins=bins, alpha=0.6, color="tomato",
              label=f"Wrong ({len(wrong_conf)})")
    ax_h.axvline(op_thresh, color="navy", linestyle="--", lw=1.5,
                 label=f"Operating threshold ({op_thresh})")
    ax_h.set_xlabel("Top Confidence Score", fontsize=12)
    ax_h.set_ylabel("Number of Photos", fontsize=12)
    ax_h.set_title(f"Confidence Distribution: Correct vs Incorrect — {TARGET_REGION}", fontsize=13)
    ax_h.legend(fontsize=10)
    ax_h.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_conf, dpi=180)
    plt.close()
    print(f"Confidence histogram saved as {output_conf}")

    # Coverage @ Precision curve
    conf_sweep = np.linspace(0, 1, 200)
    coverages, precisions_sweep = [], []
    for t in conf_sweep:
        subset = eval_df[eval_df["top_confidence"] >= t]
        coverages.append(len(subset) / len(eval_df))
        precisions_sweep.append(
            (subset["current_label"] == subset["refined_label"]).mean() if len(subset) > 0 else 1.0
        )
    fig_cov, ax_c = plt.subplots(figsize=(10, 6))
    ax_c.plot(precisions_sweep, coverages, color="steelblue", lw=2)
    idx_op = int(np.argmin(np.abs(conf_sweep - op_thresh)))
    ax_c.scatter([precisions_sweep[idx_op]], [coverages[idx_op]], color="tomato", s=80, zorder=5)
    ax_c.annotate(
        f"conf={op_thresh:.2f}\nP={precisions_sweep[idx_op]:.2f}  Coverage={coverages[idx_op]:.2f}",
        xy=(precisions_sweep[idx_op], coverages[idx_op]),
        xytext=(15, -30), textcoords="offset points", fontsize=9, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85),
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
    )
    ax_c.set_xlabel("Precision (Accuracy among labeled photos)", fontsize=12)
    ax_c.set_ylabel("Coverage (Fraction of photos labeled)", fontsize=12)
    ax_c.set_title(f"Coverage @ Precision — {TARGET_REGION}", fontsize=13)
    ax_c.set_xlim([0, 1.05])
    ax_c.set_ylim([0, 1.05])
    ax_c.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_cov, dpi=180)
    plt.close()
    print(f"Coverage @ Precision curve saved as {output_cov}")


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


def visualize_geo_temporal(csv_filename, output_geo, output_temporal):
    """Generate an interactive geographic scatter map and a monthly temporal bar chart."""
    df = pd.read_csv(csv_filename)

    # Geographic scatter map — only photos that have GPS and a refined label
    df_geo = df[
        df["location"].notna()
        & (df["location"].astype(str).str.strip() != "No GPS Data")
        & df["refined_label"].notna()
        & (df["refined_label"].astype(str).str.strip() != "")
    ].copy()

    if not df_geo.empty:
        coords = df_geo["location"].str.split(", ", expand=True)
        df_geo["lat"] = pd.to_numeric(coords[0], errors="coerce")
        df_geo["lon"] = pd.to_numeric(coords[1], errors="coerce")
        df_geo = df_geo.dropna(subset=["lat", "lon"])
        fig_geo = px.scatter_mapbox(
            df_geo, lat="lat", lon="lon",
            color="refined_label",
            hover_name="display_name",
            hover_data={"date": True, "top_confidence": ":.2%", "lat": False, "lon": False},
            title=f"Bird Photo Locations — {TARGET_REGION}",
            mapbox_style="open-street-map",
            zoom=4,
            height=700,
        )
        fig_geo.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        fig_geo.write_html(output_geo)
        print(f"Geographic map saved as {output_geo}")
    else:
        print("No geo data with labels available for map.")

    # Temporal distribution — photos per calendar month
    df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce", format="%Y-%m-%d %H:%M:%S")
    df_time = df.dropna(subset=["date_parsed"]).copy()
    if not df_time.empty:
        df_time["month"] = df_time["date_parsed"].dt.to_period("M").astype(str)
        monthly = df_time.groupby("month").size().reset_index(name="Photos")
        monthly = monthly.sort_values("month")
        fig_time = px.bar(
            monthly, x="month", y="Photos",
            title=f"Bird Photos per Month — {TARGET_REGION}",
            labels={"month": "Month", "Photos": "Number of Photos"},
            color="Photos",
            color_continuous_scale="Blues",
        )
        fig_time.update_layout(xaxis_tickangle=-45, title_font_size=16)
        fig_time.write_html(output_temporal)
        print(f"Temporal distribution chart saved as {output_temporal}")
    else:
        print("No date data available for temporal chart.")


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
    #output_csv_filename = 'US_classified_birds_report_20260530_220257.csv'
    #timestamp = '20260530_220257'
    output_csv_filename = f"{TARGET_REGION}_{base_name}_{timestamp}.csv"
    process_birds_album(detector, classifier, iqa_metric, output_csv_filename)
    visualize_species_plotly(output_csv_filename)
    visualize_geo_temporal(
        output_csv_filename,
        output_geo=f"{TARGET_REGION}_geo_map_{timestamp}.html",
        output_temporal=f"{TARGET_REGION}_temporal_{timestamp}.html",
    )
    os.makedirs("assets", exist_ok=True)
    analyze_bird_data(
        output_csv_filename,
        output_cm=f"assets/{TARGET_REGION}_confusion_matrix_{timestamp}.png",
        output_pr=f"assets/{TARGET_REGION}_precision_recall_curve_{timestamp}.png",
        output_f1=f"assets/{TARGET_REGION}_f1_by_species_{timestamp}.html",
        output_conf=f"assets/{TARGET_REGION}_confidence_histogram_{timestamp}.png",
        output_cov=f"assets/{TARGET_REGION}_coverage_precision_{timestamp}.png",
    )
    sync_keywords_from_csv(output_csv_filename)
