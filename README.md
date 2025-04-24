TGS Salt Identification Challenge Solution (BE7910)

## Overview

This repository contains the code and resources for Team 2's participation in the [TGS Salt Identification Challenge](https://www.kaggle.com/competitions/tgs-salt-identification-challenge/overview) on Kaggle, as part of the **BE7910** course project.

**Goal:** To develop a deep learning model that automatically and accurately identifies subsurface salt deposits from seismic images, using semantic segmentation techniques.

## Team Members

* Team Member 1: Marcus
* Team Member 2: Jeimy
* Team Member 3: Hannah
* Team Member 4: Mizbah
* Team Member 5: Kolin
* Team Member 6: Carlie
* Team Member 7: Asif 

## Proposed Approach

Our primary approach leverages a **U-Net based architecture** implemented in **PyTorch**. Key elements include:

* **Model:** U-Net variants with pretrained backbones (e.g., ResNet34/ResNet50).
* **Data Handling:** K-Fold Cross-Validation (stratified by salt coverage), reflection padding/resizing of input images (101x101 -> 128x128 or similar).
* **Augmentation:** Strong data augmentation using the `albumentations` library (flips, rotate, scale, shift, brightness/contrast, etc.).
* **Loss Function:** Combination losses suitable for segmentation (e.g., BCE + Dice, potentially Lovasz Hinge).
* **Optimizer/Scheduler:** Adam optimizer with learning rate scheduling (e.g., Cosine Annealing).
* **Evaluation:** Based on the official competition metric (mean Average Precision over IoU thresholds).
* **Base Code Reference:** We may adapt components from existing PyTorch solutions for this challenge (e.g., repositories like `BloodAxe/Kaggle-Salt` or similar).

## Repository Structure (Tentative)
```
├── data/                # Competition data (needs downloading)
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   ├── test/
│   │   └── images/
│   ├── train.csv
│   ├── depths.csv
│   └── sample_submission.csv
├── src/                 # Source code
│   ├── datasets.py      # PyTorch Dataset/DataLoader classes
│   ├── models.py        # Model definitions (U-Net variants)
│   ├── losses.py        # Loss function implementations
│   ├── metrics.py       # Evaluation metric implementation
│   ├── transforms.py    # Data augmentation functions
│   ├── engine.py        # Training/validation loop logic
│   ├── predict.py       # Inference script
│   ├── submission.py    # Script to generate submission file
│   └── utils.py         # Utility functions
├── configs/             # Configuration files (hyperparameters, paths)
├── notebooks/           # Jupyter notebooks for EDA, visualization
├── scripts/             # Helper scripts (e.g., data setup, pipeline execution)
├── outputs/             # Saved models, logs, predictions (add to .gitignore)
├── README.md            # This file
├── requirements.txt     # Python dependencies
└── .gitignore           # Files/directories ignored by Git
```
## Setup

1.  **Clone Repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```
2.  **Create Environment (Recommended):**
    ```bash
    conda create -n tgs-salt python=3.9 # Or desired version
    conda activate tgs-salt
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download Data:**
    * Download the competition data from [Kaggle](https://www.kaggle.com/competitions/tgs-salt-identification-challenge/data).
    * Unzip and place the data into the `data/` directory as shown in the structure above.


## Contribution Roles

* **Data Loading:** Kolin (TM5) [cite: 5]
* **Feature Engineering / Augmentation:** Marcus (TM1) [cite: 3]
* **Model Building:** Jeimy (TM2) [cite: 5]
* **Loss & Training Loop:** Hannah (TM3) [cite: 5]
* **Hyperparameter Tuning:** Carlie (TM6) [cite: 5]
* **Submission Generation:** Mizbah (TM4) [cite: 5]
* **Pipeline Integration & Reproducibility:** Asif (TM7) [cite: 5, 6]

## License

*(Consider adding an open-source license like MIT or Apache 2.0. Choose one and add the corresponding license file.)*

Example: `This project is licensed under the MIT License - see the LICENSE file for details.`