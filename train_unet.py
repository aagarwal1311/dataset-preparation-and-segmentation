import os
os.environ["WANDB_API_KEY"] = "670092a8963724063b15c169de10f964826979bb"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import wandb
import argparse
from tqdm import tqdm

from unet_model import UNet
from dataset import load_dataset

def dice_score(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    smooth = 1e-6
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()
    return (2. * intersection + smooth) / (union + smooth)

def validate(model, loader, device):
    model.eval()
    total_dice = 0
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="🔍 Validating", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            total_dice += dice_score(preds, masks).item()
    return total_dice / len(loader)

def train(args):
    wandb.init(project="coco-binary-segmentation", name="unet-binary", config=vars(args), mode="online")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, val_ds = load_dataset(image_size=args.image_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"[Epoch {epoch}]")
        for batch_idx, (imgs, masks) in loop:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

            # Log batch-wise loss
            wandb.log({
                "batch_loss": loss.item(),
                "batch": batch_idx + 1,
                "epoch": epoch
            })

        avg_loss = total_loss / len(train_loader)
        val_dice = validate(model, val_loader, device)

        # Log epoch-wise metrics
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_dice": val_dice
        })

        # Log predictions
        model.eval()
        with torch.no_grad():
            imgs, masks = next(iter(val_loader))
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)

            wandb.log({
                "Input": [wandb.Image(make_grid(imgs[:4].cpu()), caption="Input")],
                "Ground Truth": [wandb.Image(make_grid(masks[:4].cpu()), caption="Mask")],
                "Prediction": [wandb.Image(make_grid((preds[:4] > 0.5).float().cpu()), caption="Prediction")]
            })

        # Save checkpoint
        ckpt_path = f"checkpoint_epoch_{epoch}.pt"
        torch.save(model.state_dict(), ckpt_path)
        wandb.save(ckpt_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--image_size", type=tuple, default=(128, 128))
    args = parser.parse_args()

    train(args)
