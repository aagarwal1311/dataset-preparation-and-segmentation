import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import load_dataset
from unet_model import UNet
from tqdm import tqdm

def dice_score(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    smooth = 1e-6
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()
    return (2. * intersection + smooth) / (union + smooth)

def iou_score(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    smooth = 1e-6
    intersection = (preds * targets).sum()
    union = (preds + targets).clamp(0, 1).sum()
    return (intersection + smooth) / (union + smooth)

def pixel_accuracy(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    correct = (preds == targets).float().sum()
    total = torch.numel(preds)
    return correct / total

def evaluate(model, loader, device):
    model.eval()
    total_dice = 0
    total_iou = 0
    total_acc = 0
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="🔍 Evaluating"):
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)

            total_dice += dice_score(preds, masks).item()
            total_iou += iou_score(preds, masks).item()
            total_acc += pixel_accuracy(preds, masks).item()

    n = len(loader)
    print(f"\n✅ Evaluation Metrics:")
    print(f"Dice Coefficient: {total_dice / n:.4f}")
    print(f"IoU Score       : {total_iou / n:.4f}")
    print(f"Pixel Accuracy  : {total_acc / n:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="checkpoint_epoch_3.pt")
    parser.add_argument("--image_size", type=tuple, default=(128, 128))
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    _, val_ds = load_dataset(image_size=args.image_size)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # Load model
    model = UNet().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"✅ Loaded checkpoint: {args.checkpoint}")

    evaluate(model, val_loader, device)
