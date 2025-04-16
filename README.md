# COCO Binary Segmentation with U-Net

This project performs binary segmentation using a U-Net architecture on a subset of the COCO 2017 dataset (8000 images). It includes dataset downloading, annotation filtering, binary mask generation, and training a segmentation model.

## 📂 Directory Structure
coco_subset/
├── images/                   ← 8000 downloaded COCO images
├── annotations/              ← Original COCO annotations (e.g. instances_train2017.json)
├── annotations_subset.json   ← Filtered annotations for selected 8000 images
└── masks/                    ← Binary segmentation masks (PNG, 0=background, 1=object)

## Edge Cases Handled
- Skips annotations with missing/empty or invalid segmentations
- Handles overlapping polygons (merged as foreground)
- Ignores polygons with fewer than 3 points


## Implementation Overview

- `dataset_download.ipynb`: Downloads 8000 images and extracts filtered annotations.
- `dataset.py`: Loads image-mask pairs and performs preprocessing, augmentations, and train/test split.
- `ds.py`: Generates binary masks from COCO annotations and handles edge cases (empty masks, overlapping objects).
- `unet_model.py`: Defines a simple U-Net model for binary segmentation.
- `train_unet.py`: Trains U-Net using binary cross-entropy loss and logs training metrics via public Weights & Biases.
- `requirements.txt`: Lists required Python libraries.
- `python evaluate_checkpoint.py`: Evaluates the trained model checkpoint on the validation set.


## Usage Setup

1. Install dependencies:
    pip install -r requirements.txt

2. Download and prepare dataset:
    Run dataset_download.ipynb 
    #downloads 8000 images from COCO train2017. 
    #Filter annotations for selected images. 
    #Create binary PNG masks in coco_subset/masks. 
    #Handles all the edge cases.

3. Train the model:
    python train_unet.py
    #The model is a standard U-Net implemented in unet_model.py
    This includes:
        - Automatic 80/20 train/test split
        - Dice score and BCE loss computation
        - TQDM-based live progress for both training and validation
        - Visual logs and metrics in Weights & Biases
    
    Model weights and metrics are saved and tracked on the public wandb project.


4. Logging & Evaluation
    The public wandb shared logs loss, dice score, and predictions for each epoch

5. Output
    After training, predicted masks and model checkpoints are saved. 
    evaluation script: python evaluate_checkpoint.py
    Evaluation includes:
        - Dice Score
        - Binary IoU
        - Sample visualizations (input, ground truth, prediction)


