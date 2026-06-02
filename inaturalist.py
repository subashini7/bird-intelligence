import os
import requests
import pandas as pd

import time
import torch
from io import BytesIO
from PIL import Image
from transformers import pipeline

device = "mps" if torch.backends.mps.is_available() else "cpu"

def get_bird_species(place_id, place_name):
    base_url = "https://api.inaturalist.org/v1/observations/species_counts"
    params = {
        "place_id": place_id,
        "taxon_id": 3,      # 3 is the Taxon ID for Aves (Birds)
        "quality_grade": "research", # Research-grade for high data quality
        "per_page": 500     # Max results per page
    }
    
    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        print(f"Error fetching data for {place_name}")
        return []

    data = response.json()
    species_list = []
    
    for item in data['results']:
        taxon = item['taxon']
        species_list.append({
            "Place": place_name,
            "Scientific Name": taxon.get("name"),
            "Common Name": taxon.get("preferred_common_name", "N/A"),
            "Observations": item.get("count")
        })
    
    return species_list

def get_birds(regions):
    all_birds = []
    for name, pid in regions.items():
        print(f"Fetching birds for {name}...")
        all_birds.extend(get_bird_species(pid, name))
        time.sleep(1) # Respect API rate limits
    return all_birds


def download_and_process(csv_name, region, region_id, detector, output_dir, limit = 30):
    df = pd.read_csv(csv_name)
    india_birds = df[(df['Place'] == region) & (df['Observations'] > 10)]
    india_birds.sort_values('Scientific Name')
    print(f"Number of species to process: {len(india_birds)}")
    base_url = "https://api.inaturalist.org/v1/observations"
    
    headers = {
        "User-Agent": "BirdClassifierProject/1.0"
    }
    for i, row in india_birds.iterrows():
        sci_name = row['Scientific Name']
        folder_name = sci_name.replace(" ", "_")
        save_path = os.path.join(output_dir, folder_name)
        os.makedirs(save_path, exist_ok=True)
        existing_files = os.listdir(save_path)
        existing_count = len([f for f in existing_files if f.endswith('.jpg')])
        if existing_count >= limit:
            print(f"✅ Skipping {sci_name}: already has {existing_count} images.")
            continue
        needed = limit - existing_count
        print(f"🔍 {i} {sci_name}: have {existing_count}, fetching {needed} more...")

        params = {
            "taxon_name": sci_name.strip(),
            "place_id": region_id,
            "quality_grade": "research",
            "photos": "true",
            "per_page": needed * 2
        }
        
        try:
            response = requests.get(base_url, params=params, headers=headers)
            if response.status_code != 200:
                print(f"Server error {response.status_code} for {sci_name}")
                continue

            data = response.json()
            newly_saved = 0
            
            for obs in data.get('results', []):
                if (existing_count + newly_saved) >= limit:
                    break
                file_id = f"{obs['id']}.jpg"
                if file_id in existing_files:
                    continue
                img_url = obs['photos'][0]['url'].replace('square', 'large')
                img_resp = requests.get(img_url)
                img = Image.open(BytesIO(img_resp.content)).convert("RGB")
                results = detector(img)
                birds = [res for res in results if res["label"] == "bird" and res["score"] > 0.7]
                bird_count = len(birds)
                if bird_count == 1:
                    box = birds[0]["box"]
                    cropped = img.crop((box["xmin"], box["ymin"], box["xmax"], box["ymax"]))
                
                    # Resize to 224x224 for DINOv2
                    final = cropped.resize((224, 224), Image.Resampling.LANCZOS)
                    
                    # Save
                    final.save(os.path.join(save_path, f"{obs['id']}.jpg"))
                    newly_saved += 1
                
                
            print(f"✨ {sci_name} total now: {existing_count + newly_saved}/{limit}")
        
        except Exception as e:
            print(f"Error processing {sci_name}: {e}")
        time.sleep(0.5) # API courtesy

if __name__ == "__main__":
    # Define regions
    regions = {
        "India": 6683,
        "Singapore": 6734,
        "United Kingdom": 6857
    }
    all_birds = get_birds(regions)
    
    # Save to CSV
    df = pd.DataFrame(all_birds)
    df.to_csv("regional_birds.csv", index=False)
    print("Finished! Saved to regional_birds.csv")
    detector = pipeline(
        "object-detection",
        model="facebook/detr-resnet-50",
        device=device
    )
    # download_and_process('regional_birds.csv', 'India', 6683, detector, "./processed_india_birds")
    # download_and_process('regional_birds.csv', 'Singapore', 6734, detector, "./processed_singapore_birds")
    download_and_process('regional_birds.csv', 'United Kingdom', 6857, detector, "./processed_uk_birds")