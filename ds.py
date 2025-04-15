import os
import json
import random
import requests
import zipfile
from tqdm import tqdm
from pycocotools.coco import COCO

def download_file(url, dest_path):
    print(f"Downloading {url}...")
    r = requests.get(url, stream=True)
    with open(dest_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

def unzip_file(zip_path, extract_to):
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def download_image(url, path):
    try:
        img_data = requests.get(url).content
        with open(path, 'wb') as handler:
            handler.write(img_data)
    except Exception as e:
        print(f"Failed to download {url}: {e}")

def download_coco_subset(save_dir, num_images=8000):
    os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)
    annotations_dir = os.path.join(save_dir, "annotations")
    os.makedirs(annotations_dir, exist_ok=True)

    # Step 1: Download and extract annotations if not present
    ann_file_path = os.path.join(annotations_dir, "instances_train2017.json")
    if not os.path.exists(ann_file_path):
        zip_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        zip_path = os.path.join(save_dir, "annotations_trainval2017.zip")
        download_file(zip_url, zip_path)
        unzip_file(zip_path, save_dir)

    # Step 2: Load COCO annotations
    coco = COCO(ann_file_path)
    img_ids = coco.getImgIds()
    selected_ids = random.sample(img_ids, num_images)

    selected_imgs = coco.loadImgs(selected_ids)
    all_anns = []

    print(f"Downloading {num_images} images...")
    for img in tqdm(selected_imgs):
        url = img['coco_url']
        file_name = img['file_name']
        out_path = os.path.join(save_dir, "images", file_name)
        if not os.path.exists(out_path):
            download_image(url, out_path)

        anns = coco.loadAnns(coco.getAnnIds(imgIds=img['id']))
        all_anns.extend(anns)

    # Step 3: Save filtered annotations
    subset_anns = {
        "info": {},
        "licenses": [],
        "images": selected_imgs,
        "annotations": all_anns,
        "categories": coco.loadCats(coco.getCatIds())
    }

    with open(os.path.join(save_dir, "annotations_subset.json"), 'w') as f:
        json.dump(subset_anns, f)

    print("✅ Done. Saved 8000 images and annotations_subset.json")

# Direct call if running in notebook
download_coco_subset("./coco_subset", 8000)
