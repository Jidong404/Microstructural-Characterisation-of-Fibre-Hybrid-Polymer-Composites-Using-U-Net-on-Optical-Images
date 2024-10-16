import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomLoss(nn.Module):
    def __init__(self, ce_weight=0.5, dice_weight=0.5, class_weights=None, num_classes=5):
        super(CustomLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.class_weights = class_weights
        self.num_classes = num_classes
    
    def forward(self, preds, targets):

        targets = targets.squeeze()
        ce_loss = F.cross_entropy(preds, targets, weight=self.class_weights,reduction='mean')

        

        dice_loss_val = self.GDL(preds, targets,self.num_classes,self.class_weights)  
        
        # Combine the two losses using weighted sum
        loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss_val
        
        return loss
    
    
    def GDL(self, preds, targets,num_classes,class_weights):
    
        dice_losses = torch.zeros(1)

        softmax = nn.Softmax(dim=1)
        preds = softmax(preds)

        target_mask = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        intersection = (self.class_weights*(preds*target_mask).sum(dim=(-1, -2))).sum(dim=-1)

        Union = (self.class_weights*((preds**2).sum(dim=(-1, -2)) + (target_mask**2).sum(dim=(-1, -2))) ).sum(dim=-1)

        Dice_coef = (2 * intersection) / (Union + 1e-8)

        dice_losses = 1-Dice_coef

        return dice_losses.mean()