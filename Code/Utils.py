import torch
import torchvision
from Dataset import HybridDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import os
from matplotlib import pyplot as plt
import shutil
from tifffile import imwrite

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR = r'/root/autodl-tmp'

def save_checkpoint(state, epoch, save_dir=SAVE_DIR):
    filename = os.path.join(save_dir, f"model_epoch.pth")
    print(f"=> Saving checkpoint for epoch {epoch}")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
 
def get_loaders(
    train_dir,
    BF_train_mask_dir,
    HM_train_mask_dir,
    matrix_train_mask_dir,
    void_train_mask_dir,
    potting_train_mask_dir,
    val_dir,
    BF_val_mask_dir,
    HM_val_mask_dir,
    matrix_val_mask_dir,
    void_val_mask_dir,
    potting_val_mask_dir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=2,
    pin_memory=True,
):
    train_ds = HybridDataset(
        image_dir        =  train_dir,
        BF_mask_dir      =  BF_train_mask_dir,
        HM_mask_dir      =  HM_train_mask_dir,
        matrix_mask_dir  =  matrix_train_mask_dir,
        void_mask_dir    =  void_train_mask_dir,
        potting_mask_dir =  potting_train_mask_dir,
        transform        =  train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size       = batch_size,
        num_workers      = num_workers,
        pin_memory       = pin_memory,
        shuffle          = False,
    )

    val_ds = HybridDataset(
        image_dir        = val_dir,
        BF_mask_dir      = BF_val_mask_dir,
        HM_mask_dir      = HM_val_mask_dir,
        matrix_mask_dir  = matrix_val_mask_dir,
        void_mask_dir    = void_val_mask_dir,
        potting_mask_dir = potting_val_mask_dir,
        transform        = val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size      = batch_size,
        num_workers     = num_workers,
        pin_memory      = pin_memory,
        shuffle         = False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device='cuda'):
    
    num_correct = 0
    num_pixels = 0
    num_classes = 5
    dice_score = torch.zeros(num_classes).to(device)
    precision_score = torch.zeros(num_classes).to(device)
    recall_score = torch.zeros(num_classes).to(device)
    
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            y = y.squeeze()
            softmax = nn.Softmax(dim=1)
            preds = torch.argmax(softmax(model(x)), axis=1)

            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)

            for i in range(num_classes):
                preds_i = (preds == i).float()  # predictions for class i
                y_i = (y == i).float()  # ground truth for class i
                
                true_positive = (preds_i * y_i).sum()
                false_negative = y_i.sum() - true_positive
                
                dice_score[i] += ((2 * true_positive) / ((preds_i + y_i).sum() + 1e-8))
                precision_score[i] += (true_positive / (preds_i.sum() + 1e-8))
                recall_score[i] += (true_positive / (y_i.sum() + 1e-8))

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
    print(f"Dice score: {dice_score/len(loader)}")
    print(f"Average dice score: {dice_score.mean()/len(loader)}")  
    print(f"Precision score: {precision_score/len(loader)}")
    print(f"Average precision score: {precision_score.mean()/len(loader)}")
    print(f"Recall score: {recall_score/len(loader)}")
    print(f"Average recall score: {recall_score.mean()/len(loader)}")
    
    model.train()
    
    return (num_correct/num_pixels*100, 
            dice_score.mean()/len(loader), 
            precision_score.mean()/len(loader), 
            recall_score.mean()/len(loader), 
            dice_score/len(loader), 
            precision_score/len(loader), 
            recall_score/len(loader))


def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    model.eval()
    os.makedirs(folder, exist_ok=True)
    softmax = torch.nn.Softmax(dim=1)

    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds_logits = model(x)
            preds_probs = softmax(preds_logits)
            preds = torch.argmax(preds_probs, dim=1).float()

        # Iterate over the batch
        for i in range(x.shape[0]):
            # Saving input image
            imwrite(f"{folder}/input_{idx * loader.batch_size + i}.tiff", x[i].cpu().numpy().astype(np.float32))

            # Saving predicted segmentation mask
            imwrite(f"{folder}/pred_{idx * loader.batch_size + i}.tiff", preds[i].cpu().numpy().astype(np.float32))

            # Saving ground truth mask
            imwrite(f"{folder}/gt_{idx * loader.batch_size + i}.tiff", y[i].cpu().numpy().astype(np.float32))

    model.train()






