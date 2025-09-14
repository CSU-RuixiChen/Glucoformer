
# Glucoformer: Multimodal Cross-subject Glucose Forecasting with Long-term Prediction Capability

<img width="2329" height="1772" alt="image" src="https://github.com/user-attachments/assets/ad80211f-1271-45ff-a250-6a4044bb873c" />

> Paper: https://arxiv.org/abs/2405.04517
>
> Authors: Ruixi Chen, Jia Long, Yuxuan Liao, Jiyun Zheng, XXX, Hongmei Lu, Zhimin Zhang

## Abstract
**Glucoformer** is a novel model for long-term blood glucose forecasting, achieving state-of-the-art accuracy and computational efficiency (see Figure 1). Built upon a streamlined Crossformer architecture, Glucoformer introduces several key innovations:
- **Efficient Architecture:** Redundant attention modules are removed, and a lightweight Patch Linear Attention mechanism is incorporated, enabling the model to capture complex, multivariate glucose dynamics from long sequences with reduced computational cost.
- **Transfer Learning Strategy:** We pioneer a subject-level data partitioning transfer learning approach, allowing Glucoformer to generalize across diverse populations and provide reliable predictions for subjects not seen during training—unlike conventional personalized models.
- **Comprehensive Evaluation:** Glucoformer is rigorously evaluated on both the T1DMS Virtual Dataset and the OhioT1DM Real Dataset. We conduct extensive experiments, including direct comparisons with leading blood glucose forecasting models: GRU, LSTM, Transformer, Informer, DLinear, TimeXer, PatchTST, and Crossformer.
- **Transfer Learning Pipeline:** Unlike baselines trained solely on the OhioT1DM Dataset, Glucoformer leverages pre-training on the T1DMS Dataset to capture general glucose dynamics, followed by fine-tuning on the OhioT1DM Dataset for subject-specific adaptation.
- **Thorough Analysis:** Our experiments include personalized (within-subject) vs. cross-subject generalization under subject-wise partitioning, feature-fusion strategy analysis, long-horizon forecasting, and computational complexity profiling.

