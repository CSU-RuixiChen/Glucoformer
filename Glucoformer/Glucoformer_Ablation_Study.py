import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)  # 插入到搜索路径的最前面
from Glucose_Data.T1DMS_GlucoseDataset import T1DMS_GlucoseDataset
from Glucose_Data.OhioDataset import *
from Glucoformer.Glucoformer_models.Glucoformer import *
from Glucoformer.Glucoformer_models_TF_ETSA.Glucoformer import Glucoformer_TF_ETSA
from Glucoformer.Glucoformer_models_TF_RD.Glucoformer import Glucoformer_TF_RD
from Glucoformer.utils.Glucoformer_options import GlucoformerOptions
from Glucoformer.utils.train_eval import *
from torch.utils.tensorboard import SummaryWriter
from Glucoformer.utils.tools import *
import random
import argparse


if __name__ == "__main__":
    
    config = GlucoformerOptions().parse()
    
    config.model_name = "Glucoformer_RD_ETSA"
    prediction_task = [30, 60, 90]
    for pred_len_raw in prediction_task:
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        config.pred_len = pred_len_raw
        config.label_len = 2 * pred_len_raw
        config.seq_len = 4 * pred_len_raw
        # define model
        seq_len = int(config.seq_len/config.time_step)
        label_len = int(config.label_len/config.time_step)
        pred_len = int(config.pred_len/config.time_step)
        seg_len = int(config.seg_len/config.time_step)
        config.enc_in = config.data_dim
        model = Glucoformer(
            data_dim=config.data_dim, in_len=seq_len, out_len=pred_len, seg_len=seg_len, output_size=config.c_out,
            factor=config.factor, d_model=config.d_model, d_ff=config.d_ff, n_heads=config.n_heads,
            e_layers=config.e_layers, dropout=config.dropout)
        run(config, model, Tranning_mode="normal_train")
    
    config.model_name = "Glucoformer_TF_ETSA"
    prediction_task = [30, 60, 90]
    for pred_len_raw in prediction_task:
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        config.pred_len = pred_len_raw
        config.label_len = 2 * pred_len_raw
        config.seq_len = 4 * pred_len_raw
        # define model
        seq_len = int(config.seq_len/config.time_step)
        label_len = int(config.label_len/config.time_step)
        pred_len = int(config.pred_len/config.time_step)
        seg_len = int(config.seg_len/config.time_step)
        config.enc_in = config.data_dim
        model = Glucoformer_TF_ETSA(
            data_dim=config.data_dim, in_len=seq_len, out_len=pred_len, seg_len=seg_len, output_size=config.c_out,
            factor=config.factor, d_model=config.d_model, d_ff=config.d_ff, n_heads=config.n_heads,
            e_layers=config.e_layers, dropout=config.dropout)
        run(config, model, Tranning_mode="pretrain")


    config.model_name = "Glucoformer_TF_RD"
    prediction_task = [30, 60, 90]
    for pred_len_raw in prediction_task:
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        config.pred_len = pred_len_raw
        config.label_len = 2 * pred_len_raw
        config.seq_len = 4 * pred_len_raw
        # define model
        seq_len = int(config.seq_len/config.time_step)
        label_len = int(config.label_len/config.time_step)
        pred_len = int(config.pred_len/config.time_step)
        seg_len = int(config.seg_len/config.time_step)
        config.enc_in = config.data_dim
        model = Glucoformer_TF_RD(
            data_dim=config.data_dim, in_len=seq_len, out_len=pred_len, seg_len=seg_len, output_size=config.c_out,
            factor=config.factor, d_model=config.d_model, d_ff=config.d_ff, n_heads=config.n_heads,
            e_layers=config.e_layers, dropout=config.dropout)
        run(config, model, Tranning_mode="pretrain")
    