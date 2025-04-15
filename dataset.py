import os
import glob
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from tqdm import tqdm
class CocoSegmentationDataset(Dataset):
    def __init__(self, image_files, mask_files, image_size=(256, 256)):
        self.image_files = image_files
        self.mask_files = mask_files
        self.image_size = image_size

        self.img_transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor()
        ])
        self.mask_transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx]).convert("RGB")
        mask = Image.open(self.mask_files[idx]).convert("L")

        img = self.img_transform(img)
        mask = self.mask_transform(mask)
        mask = (mask > 0).float()  # binarize

        return img, mask

def load_dataset(data_dir="./coco_subset", image_size=(256, 256), test_size=0.2):
    image_dir = os.path.join(data_dir, "images")
    mask_dir = os.path.join(data_dir, "masks")

    image_files = []
    mask_files = []

    print("🔍 Filtering image-mask pairs...")
    for img_file in tqdm(sorted(glob.glob(os.path.join(image_dir, "*.jpg")))):
        base = os.path.splitext(os.path.basename(img_file))[0]
        mask_file = os.path.join(mask_dir, base + "_mask.png")

        if os.path.exists(mask_file):
            image_files.append(img_file)
            mask_files.append(mask_file)

    assert len(image_files) == len(mask_files), "Image-mask mismatch after filtering."

    img_train, img_test, mask_train, mask_test = train_test_split(
        image_files, mask_files, test_size=test_size, random_state=42)

    train_ds = CocoSegmentationDataset(img_train, mask_train, image_size)
    test_ds = CocoSegmentationDataset(img_test, mask_test, image_size)

    return train_ds, test_ds

