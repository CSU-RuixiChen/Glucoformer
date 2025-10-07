# Glucoformer: Multimodal Cross-subject Glucose Forecasting with Long-term Prediction Capability

<img src="figures/Overview of Glucoformer.png" alt="Overview of Glucoformer" />

> Paper: 
>
> Authors: Ruixi Chen, Jia Long, Yuxuan Liao, Jiyun Zheng, XXX, Hongmei Lu, Zhimin Zhang

---

## Table of Contents:
- [1. Project Contributions](#1-Project-Contributions)
- [2. The Overview Of Repository](#2-The-Overview-Of-Repository)
  - [2.1 Project Folder Structure](#21-Project-Folder-Structure)
  - [2.2 Key Components Summary](#22-Key-Components-Summary)
- [3. Usage Guide](#3-Usage-Guide)
  - [3.1 Environment Setup](#31-Environment-Setup)
  - [3.2 Dataset Preparation](#32-Dataset-Preparation)
  - [3.3 Experiment Running](#33-Experiment-Running)
- [4. Contact](#4-Contact)

---

## 1. Project Contributions
**Glucoformer** is a novel model for long-term blood glucose forecasting, achieving state-of-the-art accuracy and computational efficiency. Built upon a streamlined Crossformer architecture, Glucoformer introduces several key innovations:
- **Efficient Architecture:** Redundant attention modules are removed, and a lightweight Patch Linear Attention mechanism is incorporated, enabling the model to capture complex, multivariate glucose dynamics from long sequences with reduced computational cost.
- **Transfer Learning Strategy:** We pioneer a subject-level data partitioning transfer learning approach, allowing Glucoformer to generalize across diverse populations and provide reliable predictions for subjects not seen during training—unlike conventional personalized models.
- **Comprehensive Evaluation:** Glucoformer is rigorously evaluated on both the T1DMS Virtual Dataset and the OhioT1DM Real Dataset. We conduct extensive experiments, including direct comparisons with leading blood glucose forecasting models: GRU, LSTM, Transformer, Informer, DLinear, TimeXer, PatchTST, and Crossformer.
- **Transfer Learning Pipeline:** Unlike baselines trained solely on the OhioT1DM Dataset, Glucoformer leverages pre-training on the T1DMS Dataset to capture general glucose dynamics, followed by fine-tuning on the OhioT1DM Dataset for subject-specific adaptation.
- **Thorough Analysis:** Our experiments include personalized (within-subject) vs. cross-subject generalization under subject-wise partitioning, feature-fusion strategy analysis, long-horizon forecasting, and computational complexity profiling.

## 2. The Overview Of Repository

### 2.1 Project Folder Structure

<pre>
BGPrediction
├── Crossformer/
│   ├── Crossformer_main.py
│   ├── cross_models/
│   ├── save_Crossformer_model_*/
│   ├── save_Crossformer_prediction_*/
│   └── save_scaler/
├── DLinear/
├── GRU/
├── LSTM/
├── Informer/
├── PatchTST/
├── TimeXer/
├── Transformer/
├── Glucoformer/
│   ├── Glucoformer_main.py
│   ├── Glucoformer_models/
│   ├── utils/
│   │   ├── Glucoformer_options.py
│   │   ├── tools.py
│   │   └── train_eval.py
│   ├── save_Glucoformer_model_*/
│   ├── save_Glucoformer_prediction_*/
│   └── save_scaler/
├── Glucose_Data/
├── Experiments/
│   ├── Core.py
│   ├── Feature_Fusion_Experiment.py
│   ├── Glucoformer_optuna.py
│   ├── Diagram.ipynb
│   ├── Visualiser.ipynb
│   ├── Cross-subject_Generalization/
│   ├── feature_fusion_Glucoformer_results/
│   └── optuna_results_20250801_115557/
├── save_loss/
├── All_models_run.sh
├── requirements.txt
├── .gitignore
└── README.md
</pre>

### 2.2 Key Components Summary

#### （1）File Overview: 
- **Crossformer/Crossformer_main.py**: Main script to run the Crossformer model for blood glucose prediction.
- **Crossformer/cross_models/**: Contains the model architecture and related modules for Crossformer.
- **Crossformer/save_Crossformer_model_*** and **save_Crossformer_prediction_***: Directories for saving trained model checkpoints and prediction results, respectively.
- **Crossformer/save_scaler/**: Stores data normalization scalers used during training and inference.
- The folder structure and file usage for other baseline models (such as **DLinear**, **GRU**, **LSTM**, **Informer**, **PatchTST**, **TimeXer**, and **Transformer**) are similar to the above. Each contains a main script for running the model, a subfolder for model definitions, directories for saving model checkpoints and predictions, and a scaler directory for normalization.

- **Glucoformer/utils/Glucoformer_options.py**: Handles configuration and hyperparameter options for experiments.
- **Glucoformer/utils/train_eval.py**: Contains training and evaluation routines, including model training loops and validation.
- **Glucoformer/utils/tools.py**: Utility functions for logging loss, early stopping, and directory management.
- **Glucoformer/save_Glucoformer_model_*** and **save_Glucoformer_prediction_***: Folders for saving Glucoformer model weights and prediction outputs.
- **Glucoformer/save_scaler/**: Stores normalization scalers for Glucoformer.
- **Experiments/Core.py**: Main experimental pipeline for running personalized and cross-subject prediction experiments with Glucoformer, including model training and evaluation.
- **Experiments/Feature_Fusion_Experiment.py**: Script for running feature fusion experiments and saving results (including RMSE/MAE) to CSV files.
- **Experiments/Glucoformer_optuna.py**: Hyperparameter optimization using Optuna for Glucoformer.
- **Experiments/Visualiser.ipynb**: Jupyter notebook for visualizing and analyzing model predictions and training curves.
- **Experiments/Cross-subject_Generalization/**: Directory for cross-subject generalization experiment scripts and results.
- **Experiments/feature_fusion_Glucoformer_results/**: Stores results from feature fusion experiments.
- **Experiments/optuna_results_20250801_115557/**: Stores results from Optuna hyperparameter optimization.
- **All_models_run.sh**: Shell script to run all models or batch experiments.
- **requirements.txt**: Lists all Python dependencies required to run the project.
- **Glucose_Data/**: Contains raw and processed blood glucose datasets.
- **save_loss/**: Directories for saving training and validation loss logs.

#### （2）Model Running:  
  Main scripts such as `Glucoformer_main.py` and `Crossformer_main.py` are used to train and test models.  

#### （3）Result Saving:  
  Model weights and predictions are saved in the corresponding `save_*` directories.  

#### （4）Visualization:
  Jupyter notebooks in the `Experiments/` folder provide comprehensive tools for visualizing and analyzing model performance. 
  **Examples**:
  <div align="center">
    <img src="figures/Blood Glucose Prediction Comparison Curve.png" alt="Blood Glucose Prediction Comparison" width="80%"><br>
  </div>
Figure 1. Blood Glucose Prediction Comparison Curve: Comparison of predicted and actual blood glucose values for Patient 563 (PH=30min) using different models. The plot demonstrates the accuracy and temporal alignment of each model's predictions.

  <div align="center">
    <img src="figures/Clarke Error Grid Analysis.png" alt="Clarke Error Grid Analysis" width="80%"><br>
  </div>
Figure 2. Clarke Error Grid Analysis for Multiple Models: Clarke Error Grid Analysis (PH=30min, Patient 596) for nine models. Each subplot shows the clinical accuracy of predictions, with most points falling in the clinically acceptable zones (A and B).

## 3. Usage Guide

### 3.1 Environment Setup

To set up the environment for this project, please follow these steps:

#### (1)Clone the repository

```bash
git clone https://github.com/CSU-RuixiChen/BGPrediction.git
cd BGPrediction
```

#### (2)Create and activate a Python virtual environment

```bash
# Using venv
python3 -m venv ENVIRONMENT_NAME
source ENVIRONMENT_NAME/bin/activate

# Or using conda
conda create -n ENVIRONMENT_NAME python=3.13.5
conda activate ENVIRONMENT_NAME
```

#### (3)Install required packages

```bash
pip install -r requirements.txt
```

After these steps, your environment will be ready to run the code and experiments in this repository.


### 3.2 Dataset Preparation

The T1DMS virtual dataset and the OhioT1DM real-world dataset were utilized in this project. The T1DMS virtual dataset was employed for pre-training and development of the population-level Glucoformer model, enabling the model to learn general blood glucose dynamics across diverse virtual subjects. The OhioT1DM real-world dataset was used for fine-tuning and evaluation of the Glucoformer model, as well as for training and evaluation of other baseline models. It also served as the benchmark for assessing final model performance on real patient data. Due to file size and data sharing restrictions, the original datasets are not included in this repository. However, the complete datasets can be obtained from the following official sources:

- **T1DMS Virtual Dataset**: The T1DMS dataset was generated using the UVa/Padova simulator (available at [https://tegvirginia.com/t1dms/
]), and we provide the scenario files for scenarios 1–60 to facilitate replication.
- **OhioT1DM Real Dataset**: https://webpages.charlotte.edu/rbunescu/ohiot1dm.html

After obtaining the data, please place it in a new subfolder following the directory structure below. This will ensure compatibility with the data processing scripts, allowing you to generate the processed data files required for model development and replication.

- **OhioT1DM Raw Dataset:**
  <pre>
  Glucose_Data/OhioT1DM_processed_dataset/OhioT1DM_raw_dataset/
  ├── OhioT1DM-2018-training/
  │   ├── 559-ws-training.xml
  │   ├── 563-ws-training.xml
  │   ├── 570-ws-training.xml
  │   ├── 575-ws-training.xml
  │   ├── 588-ws-training.xml
  │   └── 591-ws-training.xml
  ├── OhioT1DM-2018-testing/
  │   ├── 559-ws-testing.xml
  │   ├── 563-ws-testing.xml
  │   ├── 570-ws-testing.xml
  │   ├── 575-ws-testing.xml
  │   ├── 588-ws-testing.xml
  │   └── 591-ws-testing.xml
  ├── OhioT1DM-2020-training/
  │   ├── 540-ws-training.xml
  │   ├── 544-ws-training.xml
  │   ├── 552-ws-training.xml
  │   ├── 567-ws-training.xml
  │   ├── 584-ws-training.xml
  │   └── 596-ws-training.xml
  └── OhioT1DM-2020-testing/
      ├── 540-ws-testing.xml
      ├── 544-ws-testing.xml
      ├── 552-ws-testing.xml
      ├── 567-ws-testing.xml
      ├── 584-ws-testing.xml
      └── 596-ws-testing.xml
  </pre>

- **T1DMS Raw Dataset:**

  <pre>
  Glucose_Data/T1DMS_GlucoseDataset/
  ├── sim_data_train_dataset.mat
  ├── sim_data_validate_dataset.mat
  ├── sim_results_train_dataset.mat
  └── sim_results_validate_dataset.mat
  </pre>

### 3.3 Experiment Running

#### (1)Train and Test a Single Model

- To train and test an individual model (e.g., Glucoformer or Crossformer), run the corresponding main script:
  - Glucoformer:
    ```bash
    python Glucoformer/Glucoformer_main.py
    ```
  - Crossformer:
    ```bash
    python Crossformer/Crossformer_main.py
    ```
- You can customize model parameters and data paths within the main scripts for personalized experiments.

#### (2)Batch Training of All Models

- To train and test all models in batch, simply run the shell script:
  ```bash
  bash All_models_run.sh
  ```
- This script will automatically execute each model's main program for comprehensive benchmarking.

#### (3)Feature Fusion and Validation Experiments

- For feature fusion experiments, use:
  ```bash
  python Experiments/Feature_Fusion_Experiment.py
  ```
- This script fuses multiple features, trains models, and saves results (e.g., RMSE/MAE) to CSV files for further analysis.

#### (4)Hyperparameter Optimization

- To optimize Glucoformer hyperparameters, run the Optuna script:
  ```bash
  python Experiments/Glucoformer_optuna.py
  ```
- This script automatically searches for the best parameter combinations and saves optimization results.

#### (5)Results Visualization and Analysis

- After training, use the Jupyter notebook for visualization and analysis:
  - Open and run:
    ```
    Experiments/Visualiser.ipynb
    ```
- The notebook provides tools for plotting prediction curves, error analysis, and more, helping you interpret model performance.

---

By following these steps, you can flexibly conduct single-model training, batch experiments, feature fusion, hyperparameter optimization, and result visualization to comprehensively evaluate and improve blood glucose prediction models.

## 4. Contact

- **Ruixi Chen** at [242311059@csu.edu.cn](242311059@csu.edu.cn)

