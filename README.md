# Glucoformer: Multimodal Cross-subject Glucose Forecasting with Long-term Prediction Capability

<img width="2329" height="1772" alt="image" src="https://github.com/user-attachments/assets/ad80211f-1271-45ff-a250-6a4044bb873c" />

> Paper: 
>
> Authors: Ruixi Chen, Jia Long, Yuxuan Liao, Jiyun Zheng, XXX, Hongmei Lu, Zhimin Zhang

---

Table of Contents:
- [1. Project Contributions](#1-Abstract)
- [2. The overview of repository](#2-The-overview-of-repository)
  - [2.1 Project Folder Structure](#21-Project-Folder-Structure)
  - [2.2 Key Components Summary](#22-Key-Components-Summary)
- [3. How to work with the repository?](#2-how-to-work-with-the-repository)
  - [2.1 Setting up the environment](#21-setting-up-the-environment)
  - [2.2 Running the experiment](#22-running-the-experiment)
  - [2.3 Downloading the data](#23-loading-the-data)
- [3. How to cite](#3-how-to-cite)

---

## 1. Project Contributions
**Glucoformer** is a novel model for long-term blood glucose forecasting, achieving state-of-the-art accuracy and computational efficiency. Built upon a streamlined Crossformer architecture, Glucoformer introduces several key innovations:
- **Efficient Architecture:** Redundant attention modules are removed, and a lightweight Patch Linear Attention mechanism is incorporated, enabling the model to capture complex, multivariate glucose dynamics from long sequences with reduced computational cost.
- **Transfer Learning Strategy:** We pioneer a subject-level data partitioning transfer learning approach, allowing Glucoformer to generalize across diverse populations and provide reliable predictions for subjects not seen during training—unlike conventional personalized models.
- **Comprehensive Evaluation:** Glucoformer is rigorously evaluated on both the T1DMS Virtual Dataset and the OhioT1DM Real Dataset. We conduct extensive experiments, including direct comparisons with leading blood glucose forecasting models: GRU, LSTM, Transformer, Informer, DLinear, TimeXer, PatchTST, and Crossformer.
- **Transfer Learning Pipeline:** Unlike baselines trained solely on the OhioT1DM Dataset, Glucoformer leverages pre-training on the T1DMS Dataset to capture general glucose dynamics, followed by fine-tuning on the OhioT1DM Dataset for subject-specific adaptation.
- **Thorough Analysis:** Our experiments include personalized (within-subject) vs. cross-subject generalization under subject-wise partitioning, feature-fusion strategy analysis, long-horizon forecasting, and computational complexity profiling.

## 2. The overview of repository

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
│   ├── optuna_results_20250801_115557/
│   └── __pycache__/
├── save_loss/
├── All_models_run.sh
├── requirements.txt
├── .gitignore
└── README.md
</pre>

### 2.2 Key Components Summary

#### Brief introduction of the files: 
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

#### Model Running:  
  Main scripts such as `Glucoformer_main.py` and `Crossformer_main.py` are used to train and test models.  

#### Result Saving:  
  Model weights and predictions are saved in the corresponding `save_*` directories.  

#### Visualization:
  Jupyter notebooks in the `Experiments/` folder provide comprehensive tools for visualizing and analyzing model performance. 
  - **Examples**:
  <div align="center">
    <img src="figures/Blood Glucose Prediction Comparison Curve.png" alt="Blood Glucose Prediction Comparison" width="80%"><br>
  </div>
Figure 1. Blood Glucose Prediction Comparison Curve： Comparison of predicted and actual blood glucose values for Patient 563 (PH=30min) using different models. The plot demonstrates the accuracy and temporal alignment of each model's predictions.

  <div align="center">
    <img src="figures/Clarke Error Grid Analysis.png" alt="Clarke Error Grid Analysis" width="80%"><br>
  </div>
Figure 2. Clarke Error Grid Analysis for Multiple Models： Clarke Error Grid Analysis (PH=30min, Patient 596) for nine models. Each subplot shows the clinical accuracy of predictions, with most points falling in the clinically acceptable zones (A and B).



