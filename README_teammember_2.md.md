# TGS Salt Identification Challenge - 
Team Member 2: Jeimy Martinez
Team member 2: Build and evaluate predictive models

## Model Architectures Tried
I implemented a U-Net architecture using two different backbones:
- ResNet34 (pretrained)
- ResNet50 (pretrained)

Both architectures were tested using two different loss functions:
- BCE + Dice Combined Loss
- Lovasz Hinge Loss

## Training Settings
- Optimizer: Adam
- Learning Rate: 1e-4
- Batch Size: 64
- Number of Epochs: 20
- Data Augmentation: Horizontal flip, vertical flip, random rotate, brightness/contrast adjustment

## Filtering
- Only images with salt masks (non-empty masks) were used for training (following top solutions approach).

## Final Results Summary

| Experiment | Backbone | Loss Function | Final Train Loss | Final Val Loss | Final Val IoU |
|:-----------|:---------|:--------------|:-----------------|:---------------|:--------------|
| Exp1       | ResNet34 | BCE + Dice    | 0.2203            | 0.2764         | 0.7131        |
| Exp2       | ResNet34 | Lovasz        | 0.3070            | 0.3134         | 0.7609        |
| Exp3       | ResNet50 | BCE + Dice    | 0.1800            | 0.2669         | 0.7624        |
| Exp4       | ResNet50 | Lovasz        | 0.2648            | 0.2895         | 0.7694        |

## Best Performing Model
- **UNet(ResNet50) + Lovasz Loss**
- **Validation IoU: 0.7694**
- This model achieved the best overall validation performance across all experiments.

## Folder Structure
- `models/` : Trained model checkpoints (.pth files)
- `results/` : Training curves (Loss/IoU) and final results table (CSV)

---
Date: April 27, 2025
