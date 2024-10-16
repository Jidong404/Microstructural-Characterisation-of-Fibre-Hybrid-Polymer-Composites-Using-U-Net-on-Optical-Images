import torch
import albumentations as A
import shutil
import cv2
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm #this is for the progress bar
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from U_Net import UNET
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.optim.lr_scheduler as lr_scheduler
import os
from loss_customised_fn import CustomLoss
from Utils import(
  load_checkpoint, 
  save_checkpoint,
  get_loaders,
  check_accuracy,
  save_predictions_as_imgs,
)
############################################

LEARNING_RATE = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_EPOCHS = 200



NUM_WORKERS = 4
IMAGE_HEIGHT = 256 
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False
SAVE_DIR = r'path/save'
#############################################
#tensorboard --logdir=runs
#dir and path 
train_dir = r'Train/img'
BF_mask_dir = r'Train/mask1'
HM_mask_dir = r'Train/mask2'
matrix_mask_dir = r'Train/mask3'
void_mask_dir = r'Train/mask4'
potting_mask_dir = r'Train/mask5'

val_dir = r'Val/img'
BF_val_mask_dir = r'Val/BF'
HM_val_mask_dir = r'Val/HM'
matrix_val_mask_dir = r'Val/Matrix'
void_val_mask_dir = r'Val/Void'
potting_val_mask_dir = r'Val/Potting'
#############################################


def get_gradient_norms(model):
    gradient_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradient_norms[name] = param.grad.norm().item()
    return gradient_norms

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def train_fn(loader, model, optimizer, loss_fn, scaler):
    
    loop = tqdm(loader) 
    losses = []  
    
    for batch_idx, (data, targets) in enumerate(loop):
        
        data = data.to(device=DEVICE)
        targets = targets.type(torch.long)
        targets = targets.to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)


        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        
        # append current batch loss
        losses.append(loss.item())

        


    # return average loss
    return sum(losses) / len(losses)

#val loss
def validate_fn(loader, model, loss_fn):
    
    loop = tqdm(loader)
    losses = []
    model.eval()

    with torch.no_grad():
        for _, (data, targets) in enumerate(loop):

            data = data.to(device=DEVICE)
            targets = targets.type(torch.long).to(device=DEVICE)
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            losses.append(loss.item())

    return sum(losses) / len(losses)


def main():
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.2),
            A.VerticalFlip(p=0.2),
            A.Transpose(p=0.2),
            A.Rotate(limit=45, p=0.2),
            A.Blur(blur_limit=3, p=0.2),
            A.GaussNoise(var_limit=(50.0, 100.0), p=0.2),
            A.RandomBrightnessContrast(p=0.4),
            A.RandomGamma(p=0.4),
            A.Normalize(mean=(0.0,), std=(1.0,), max_pixel_value=255.0),

            ToTensorV2(transpose_mask=True),
        ]
    )

    val_transforms = A.Compose(
        [
            A.HorizontalFlip(p=0.2),
            A.VerticalFlip(p=0.2),
            A.Transpose(p=0.2),
            A.Blur(blur_limit=3, p=0.2),
            A.GaussNoise(var_limit=(50.0, 100.0), p=0.2),
            A.RandomBrightnessContrast(p=0.4),
            A.RandomGamma(p=0.4),
            A.Normalize(mean=(0.0,), std=(1.0,), max_pixel_value=255.0),

            ToTensorV2(transpose_mask=True),
        ]
    )

    model = UNET(in_channels=1, out_channels=5).to(DEVICE)
    initialize_weights(model)

   
    class_weights = torch.tensor([311157.0, 543513.0, 3419890.0,  380504.0, 4967371.0], device=DEVICE)
    class_weights = 1/(class_weights)
    class_weights /= class_weights.sum() 
    loss_fn = loss_fn = CustomLoss(ce_weight=1, dice_weight=0, class_weights=class_weights, num_classes=5)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)



    train_loader, val_loader = get_loaders(
            train_dir, 
            BF_mask_dir,
            HM_mask_dir,
            matrix_mask_dir,
            void_mask_dir,
            potting_mask_dir, 
            val_dir,
            BF_val_mask_dir,
            HM_val_mask_dir,
            matrix_val_mask_dir,
            void_val_mask_dir,
            potting_val_mask_dir,
            BATCH_SIZE,
            train_transform,
            val_transforms,
            NUM_WORKERS,
            PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)


    check_accuracy(val_loader, model, device = DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    writer = SummaryWriter()

    best_dice_score = 0.0  # Initialise the best dice score


    for epoch in range(NUM_EPOCHS):
        print('the epoch num is', epoch)
        ave_train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        ave_val_loss = validate_fn(val_loader, model, loss_fn)
        # log the average losses to tensorboard
        writer.add_scalar('Training loss', ave_train_loss, epoch)
        writer.add_scalar('Validation loss', ave_val_loss, epoch)

        train_accuracy, train_avg_dice, train_precision, train_recall, train_dice_class, train_precision_class, train_recall_class = check_accuracy(train_loader, model, device=DEVICE)
        val_accuracy, val_avg_dice, val_precision, val_recall, val_dice_class, val_precision_class, val_recall_class = check_accuracy(val_loader, model, device=DEVICE)

        writer.add_scalar('Training Accuracy', train_accuracy, epoch)
        writer.add_scalar('Training Average Dice Score', train_avg_dice, epoch)
        writer.add_scalar('Training Precision', train_precision, epoch)
        writer.add_scalar('Training Recall', train_recall, epoch)
        
        
        writer.add_scalar('Validation Accuracy', val_accuracy, epoch)
        writer.add_scalar('Validation Average Dice Score', val_avg_dice, epoch)
        writer.add_scalar('Validation Precision', val_precision, epoch)
        writer.add_scalar('Validation Recall', val_recall, epoch)
                
        if val_avg_dice > best_dice_score:
            best_dice_score = val_avg_dice
            checkpoint = {
                "epoch": epoch,
                "Dice_class": val_dice_class,
                "Precision_class": val_precision_class,
                "Recall_class": val_recall_class,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, epoch, save_dir=SAVE_DIR)

            # Overwrite the previous best saved predictions
            save_predictions_as_imgs(
                val_loader, model, folder=f"saved_pred_val_new/", device=DEVICE
            )

    shutil.make_archive('saved_pred_val_new', 'zip', 'saved_pred_val_new')
    writer.close()

if __name__ == "__main__":
        main()
