

# **Wi-Fi CSI Human Activity Recognition System — README**


## 📖 Overview
--------
This project implements a deep learning pipeline for Human Activity Recognition
(HAR) using Wi-Fi Channel State Information (CSI) amplitude data. It detects
9 physical activities by analyzing how a person's movements affect Wi-Fi signals
in their environment.
```text
Supported Activities:
  0 - Nothing (Static Room)
  1 - Walking
  2 - Rotation
  3 - Jump
  4 - Wave
  5 - Lie Down
  6 - Pick Up
  7 - Sit Down
  8 - Stand Up
```
--------------------------------------------------------------------------------

PROJECT FILES
-------------
```text
  Model.py       — Neural network architecture (AmplitudeOnlyModel + Attention)
  Utils.py       — PyTorch Dataset class and CSI preprocessing logic
  Preprocess.py  — Standalone preprocessing utilities (normalization, unwrapping)
  Train1.py      — Training script with metrics logging and resource monitoring
  Evaluate.py    — Evaluation script with full metrics and confusion matrix
  Classify.py    — Real-time inference script for single .mat CSI files
```
--------------------------------------------------------------------------------
ML PIPELINE
-----------
The end-to-end machine learning pipeline consists of 5 stages:
 ```text
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                        ML PIPELINE OVERVIEW                             │
  └─────────────────────────────────────────────────────────────────────────┘
 
  [1. RAW DATA]
       WiMANS Dataset
       ├── annotation.csv      → activity labels per sample
       ├── wifi_csi/amp/*.npy  → pre-extracted CSI amplitude arrays
       └── wifi_csi/mat/*.mat  → raw MATLAB CSI files (for inference)
 
          ↓  Utils.py / CSIDataset.__getitem__()
 
  [2. PREPROCESSING]
       Per sample, applied at load time:
       ├── Differential CSI      → diff = amp[t] - amp[t-1]
       │                           highlights motion, removes static noise
       ├── Z-Score Normalization → (diff - mean) / (std + 1e-6)
       │                           zero-centers and scales each sample
       └── Pad / Truncate        → fixed shape: (30 subcarriers × 1000 time steps)
 
          ↓  Train1.py / DataLoader
 
  [3. TRAINING]
       ├── Dataset split: 80% train / 20% validation (random_split)
       ├── Sampler: WeightedRandomSampler — balances minority classes per batch
       ├── Optimizer: AdamW  (lr=5e-4, weight_decay=0.01)
       ├── Loss:      CrossEntropyLoss
       ├── Epochs:    50
       └── Batch size: 32
 
          ↓  Model.py / AmplitudeOnlyModel
 
  [4. MODEL — AmplitudeOnlyModel]
       Input: [Batch, 30, 1000]
         │
         ├── Stem        Conv1d(30→64, k=7) → BN → ReLU → MaxPool(4)
         │                                                        [B, 64, 250]
         ├── Attention   CSI_Attention (Q/K/V conv, learnable gamma)
         │                                                        [B, 64, 250]
         ├── Features    Conv1d(64→128, k=5) → BN → ReLU → MaxPool(4)
         │               Conv1d(128→256, k=3) → BN → ReLU → AdaptiveAvgPool
         │                                                        [B, 256, 1]
         └── Classifier  Flatten → Dropout(0.5) → Linear(256→9)
                                                                  [B, 9]
 
          ↓  Evaluate.py
 
  [5. EVALUATION & INFERENCE]
       Batch evaluation (Evaluate.py):
       ├── Accuracy   — overall correct predictions
       ├── Precision  — macro-averaged across all 9 classes
       ├── Recall     — macro-averaged across all 9 classes
       ├── F1-Score   — macro-averaged across all 9 classes
       └── Confusion Matrix — full 9×9 class breakdown
 
       Real-time inference (Classify.py):
       ├── Load single .mat file
       ├── Apply same Differential CSI + Z-Score preprocessing
       ├── Forward pass → softmax → argmax
       └── Output: predicted activity label + confidence score (%)
 
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  RAW CSI → PREPROCESS → TRAIN → MODEL → EVALUATE / CLASSIFY            │
  │  (Utils)    (Utils)    (Train1)  (Model)  (Evaluate / Classify)         │
  └─────────────────────────────────────────────────────────────────────────┘
```
--------------------------------------------------------------------------------
ENVIRONMENT & REQUIREMENTS
--------------------------
The following package versions were used in development and are confirmed working.
CUDA 12.1 is required for GPU-accelerated training (torch+cu121 build).
```text
  Package             Version
  ------------------- ------------
  torch               2.5.1+cu121
  torchaudio          2.5.1+cu121
  torchvision         0.20.1+cu121
  numpy               2.4.3
  pandas              3.0.2
  scipy               1.17.1
  scikit-learn        1.8.0
  matplotlib          3.10.9
  tqdm                4.67.3
  psutil              7.2.2
  colorama            0.4.6
  contourpy           1.3.3
  cycler              0.12.1
  filelock            3.25.2
  fonttools           4.62.1
  fsspec              2026.2.0
  Jinja2              3.1.6
  joblib              1.5.3
  kiwisolver          1.5.0
  MarkupSafe          3.0.3
  mpmath              1.3.0
  networkx            3.6.1
  packaging           26.2
  pillow              12.1.1
  pyparsing           3.3.2
  python-dateutil     2.9.0.post0
  setuptools          70.2.0
  six                 1.17.0
  sympy               1.13.1
  threadpoolctl       3.6.0
  typing_extensions   4.15.0
  tzdata              2026.2
  pip                 25.0.1
```
To install PyTorch with CUDA 12.1 support:

  pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
      --index-url https://download.pytorch.org/whl/cu121

To install all remaining dependencies:

  pip install numpy pandas scipy scikit-learn matplotlib tqdm psutil

CPU-only alternative (no GPU):

  pip install torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/cpu
  pip install numpy pandas scipy scikit-learn matplotlib tqdm psutil

--------------------------------------------------------------------------------

QUICK START
-----------
Follow these steps to go from zero to a trained model as fast as possible.

  Step 1 — Clone this repository

    git clone https://github.com/<your-username>/<your-repo>.git
    cd <your-repo>

  Step 2 — Install PyTorch (CUDA 12.1)

    pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
        --index-url https://download.pytorch.org/whl/cu121

  Step 3 — Install remaining dependencies

    pip install -r requirements.txt

  Step 4 — Download the WiMANS dataset from Kaggle

    pip install kaggle
    kaggle datasets download -d sharmmoh/wimans
    unzip wimans.zip -d ./dataset/

  Step 5 — Update dataset paths in Train1.py and Evaluate.py

    csv_path = "./dataset/annotation.csv"
    mat_path  = "./dataset/wifi_csi/amp"

  Step 6 — Train the model

    python Train1.py

  Step 7 — Evaluate the trained model

    python Evaluate.py

  Step 8 — Classify a single .mat file in real-time

    python Classify.py
    > Enter .mat file path: ./dataset/wifi_csi/mat/act_1_1.mat

--------------------------------------------------------------------------------

DATA SOURCE
-----------
This project uses the WiMANS dataset — the first WiFi-based multi-user activity
sensing benchmark, based on Wi-Fi Channel State Information (CSI).
```text
  Name    : WiMANS (WiFi-based Multi-user Activity Sensing)
  Samples : 11,286 CSI samples (3-second each, dual-band 2.4 / 5 GHz)
  Content : Up to 5 users performing activities simultaneously
  Annotations: User identities, locations, and activities per sample

  Kaggle  : https://www.kaggle.com/datasets/sharmmoh/wimans
  GitHub  : https://github.com/huangshk/WiMANS
```
--------------------------------------------------------------------------------

HOW TO DOWNLOAD THE DATASET
----------------------------
Option A — Kaggle (Recommended)

  1. Create a free account at https://www.kaggle.com
  2. Go to: https://www.kaggle.com/datasets/sharmmoh/wimans
  3. Click the "Download" button (downloads as a .zip archive)
  4. Extract the zip to a folder of your choice, e.g.:
       D:/WIMANS/

  Or use the Kaggle CLI (faster for large datasets):

     pip install kaggle
     kaggle datasets download -d sharmmoh/wimans
     unzip wimans.zip -d D:/WIMANS/

Option B — GitHub

  git clone https://github.com/huangshk/WiMANS.git

--------------------------------------------------------------------------------

DATASET DIRECTORY STRUCTURE
----------------------------
After extraction, the dataset folder should look like this:

 ## 📁 Dataset Directory Structure

After extraction, the dataset folder should look like this:

```text
dataset/
├── annotation.csv              # Labels: user identities, locations, activities
└── wifi_csi/
    ├── mat/
    │   ├── act_1_1.mat         # Raw CSI sample (MATLAB format)
    │   ├── act_1_2.mat
    │   └── ...                 # ~11,286 total .mat files
    ├── amp/
    │   ├── act_1_1.npy         # Pre-extracted CSI amplitude (NumPy format)
    │   ├── act_1_2.npy
    │   └── ...                 # ~11,286 total .npy files
    └── video/
        ├── act_1_1.mp4         # Synchronized reference video
        ├── act_1_2.mp4
        └── ...                 # ~11,286 total .mp4 files


  Files used by this project:
    - annotation.csv              → activity labels for training/evaluation
    - wifi_csi/amp/*.npy          → amplitude data for Train1.py and Evaluate.py
    - wifi_csi/mat/*.mat          → raw CSI data for Classify.py (real-time use)
```
--------------------------------------------------------------------------------

DATASET SETUP
-------------
After downloading, update the paths in Train1.py and Evaluate.py to point to
your local dataset location:

  csv_path = "D:/WIMANS/dataset/annotation.csv"
  mat_path  = "D:/WIMANS/dataset/wifi_csi/amp"

For Classify.py, provide the path to any individual .mat file when prompted:

  Enter .mat file path: D:/WIMANS/dataset/wifi_csi/mat/act_1_1.mat

--------------------------------------------------------------------------------

HOW TO RUN
----------

1. TRAIN THE MODEL
   Run Train1.py to train from scratch over 50 epochs.

     python Train1.py

   Output:
   - wifi_model.pth              ← Saved model weights
   - training_os_metrics.png     ← Hardware resource usage plots

   Training uses an 80/20 train/validation split and WeightedRandomSampler to
   handle class imbalance.

2. EVALUATE THE MODEL
   Run Evaluate.py against the full dataset to compute final metrics.

     python Evaluate.py

   Output includes:
   - Accuracy, Precision, Recall, F1-Score (macro)
   - Class distribution in the test set
   - Full confusion matrix

   Requires wifi_model.pth to be present in the working directory.

3. CLASSIFY A SINGLE FILE
   Run Classify.py to predict the activity in a single raw .mat CSI file.

     python Classify.py

   When prompted, enter the full path to a .mat file:

     Enter .mat file path: path/to/your/file.mat

   Output displays the detected action and confidence score.
   Type 'exit' to quit the program.

--------------------------------------------------------------------------------

MODEL ARCHITECTURE
------------------
AmplitudeOnlyModel (Model.py)
```text
  Input:  [Batch, 30 subcarriers, 1000 time steps]

  Stem Block:
    Conv1d(30 → 64, kernel=7) → BatchNorm → ReLU → MaxPool(4)

  Self-Attention:
    CSI_Attention module (query/key/value convolutions with learnable gamma)

  Feature Extractor:
    Conv1d(64 → 128, kernel=5) → BatchNorm → ReLU → MaxPool(4)
    Conv1d(128 → 256, kernel=3) → BatchNorm → ReLU → AdaptiveAvgPool → [256]

  Classifier:
    Dropout(0.5) → Linear(256 → 9)
```
--------------------------------------------------------------------------------

PREPROCESSING PIPELINE (Utils.py)
----------------------------------
Applied per sample during data loading:

  1. Differential CSI   — Subtract the previous time step to isolate motion
                          and suppress static background noise.

  2. Z-Score Normalization — Normalize the delta signal to zero mean and unit
                             variance for stable gradient flow.

  3. Padding / Truncation  — All samples are resized to (30 subcarriers × 1000
                             time steps) for consistent batch dimensions.

--------------------------------------------------------------------------------

TRAINING CONFIGURATION
-----------------------
  Optimizer    : AdamW (lr=5e-4, weight_decay=0.01)
  Loss         : CrossEntropyLoss
  Epochs       : 50
  Batch Size   : 32
  Sampler      : WeightedRandomSampler (balances minority classes per batch)
  Device       : Auto-detected (CUDA if available, else CPU)

--------------------------------------------------------------------------------

HARDWARE MONITORING (Train1.py)
-------------------------------
At the end of each epoch, the following OS-level metrics are logged and plotted:

  - Memory Usage  (RSS in MB)
  - CPU Load      (%)
  - Disk I/O      (Total MB read)
  - Batch Latency (ms/batch)

Plots are saved to: training_os_metrics.png

--------------------------------------------------------------------------------

RESULTS (EXAMPLE)
-----------------
The following output was recorded from an actual training run on this system.

  Hardware : NVIDIA GPU (CUDA 12.1), CPU load ~12.3%
  Dataset  : WiMANS — 11,286 samples, 9 activity classes
  Split    : 80% train / 20% validation

  --- Epoch 1 of 50 (first epoch, model still converging) ---

    Epoch [1/50], Batch [119/119]
    Progress  : 100.00%
    Loss      : 2.2453

    OS Metrics at epoch end:
      Memory  : 1,056.98 MB RSS
      CPU     : 12.3% load
      Disk    : 105,668.28 MB total read
      Latency : 42.54 ms/batch  (~2.94 batches/sec)
      Time    : 00:40:00 for 119 batches

    Epoch 1 Summary:
      Average Loss  : 2.2381
      Accuracy      : 11.89%
      Error Rate    : 88.11%
      Precision     : 0.1182
      Recall        : 0.1182
      F1-Score      : 0.1173

  Note: Epoch 1 accuracy near ~11% is expected for a 9-class problem (random
  baseline = 11.1%). The model learns progressively over all 50 epochs as the
  loss decreases and the weighted sampler balances class exposure across batches.

  Outputs produced after full training:
    wifi_model.pth            ← Saved model weights (~3 MB)
    training_os_metrics.png   ← 4-panel hardware resource plot

--------------------------------------------------------------------------------

NOTES
-----
- wifi_model.pth must exist before running Evaluate.py or Classify.py.
- Classify.py reads raw .mat files (MATLAB format) directly from the original
  WIMANS dataset structure; it does not use pre-extracted .npy files.
- Preprocess.py contains standalone utilities (min-max normalization, phase
  unwrapping, Gaussian encoding) that are separate from the training pipeline
  and can be reused independently.

================================================================================
