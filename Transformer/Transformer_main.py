import sys
import os
# 设置项目根目录路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # BGPredition
sys.path.insert(0, PROJECT_ROOT)
# 添加Glucoformer目录到Python路径
GLUCOFORMER_DIR = os.path.join(PROJECT_ROOT, "Glucoformer")
sys.path.insert(0, GLUCOFORMER_DIR)  # 这样 Glucoformer 中的模块可以找到
# 添加Glucoformer的父目录到Python路径
sys.path.insert(0, os.path.dirname(GLUCOFORMER_DIR))  # 这样可以导入 Glucoformer

from Glucose_Data.T1DMS_GlucoseDataset import T1DMS_GlucoseDataset
from Glucose_Data.OhioDataset import *
from Glucoformer.utils.train_eval import *
from Glucoformer.utils.tools import *
import random
import os
import argparse
# from Transformer.model.Transformer import *
   
def get_Transformer_parser():

    parser = argparse.ArgumentParser(description='Transformer')

    # forecasting task
    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    parser.add_argument('--model_name', type=str, default='Transformer', help='model name')
    parser.add_argument('--path_to_save_scaler', type=str, default="save_scaler/", help='device ids of multile gpus')
    parser.add_argument('--path_to_save_loss', type=str, default="../save_loss/", help='path to save loss')
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')

    # dataset and dataloader
    parser.add_argument('--stride', type=int, default=1, help='the step ofstride window')
    parser.add_argument('--seq_len', type=int, default=360, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=180, help='start token length')
    parser.add_argument('--pred_len', type=int, default=90, help='prediction sequence length')
    parser.add_argument('--time_step', type=int, default=5, help='sensor_sampling')
    
    # model define
    parser.add_argument('--enc_in', type=int, default=3, help='encoder input size') # train_dataset.features_num
    parser.add_argument('--dec_in', type=int, default=3, help='decoder input size') # train_dataset.features_num
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--d_model', type=int, default=256, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=3, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=512, help='dim_feedforward dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    
    # optimization
    parser.add_argument('--batch_size', type=int, default=256, help='batch size of train input data')
    parser.add_argument('--train_epochs', type=int, default=50, help='train epochs')
    parser.add_argument('--patience', type=int, default=4, help='early stopping patience')
    parser.add_argument('--pre_lr', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--ft_lr', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--step_size', type=int, default=3, help='Period of learning rate decay')
    parser.add_argument('--pre_gamma', type=float, default=0.5, help='Multiplicative factor of learning rate decay')
    parser.add_argument('--ft_gamma', type=float, default=0.75, help='Multiplicative factor of learning rate decay')
    parser.add_argument('--tolerance', type=float, default=1, help='tolerance')
    parser.add_argument('--save_pred', type=bool, default=True, help='whether to save the predicted future MTS')

    # GPU
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Device to use for computation: 'cpu' or 'cuda'")
    parser.add_argument('--best_pretrain_model', type=str, default=None, help='best_pretrain_model, eq: best_pretrain_pred(90min)_8.pth')
    parser.add_argument('--best_model', type=str, default=None, help='best_pretrain_model, eq: best_train_pred(60min)_12.pth')

    return parser


if __name__ == "__main__":
        
    parser = get_Transformer_parser()
    config = parser.parse_args()

    Seeds = [2023, 2024, 2025]
    for seed in Seeds:
        config.seed = seed
        prediction_task = [30, 60, 90]
        for pred_len_raw in prediction_task:
            random.seed(config.seed)
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)
            config.pred_len = pred_len_raw
            config.label_len = 2 * pred_len_raw
            config.seq_len = 4 * pred_len_raw
            # define model
            seq_len = int(config.seq_len / config.time_step)
            label_len = int(config.label_len / config.time_step)
            pred_len = int(config.pred_len / config.time_step)
            from Transformer_model import *
            model = Transformer_Model(input_size=config.enc_in, pred_length=pred_len, d_model=config.d_model, device=config.device, nhead=config.n_heads, 
                                    num_encoder_layers=config.e_layers, num_decoder_layers=config.d_layers, dim_feedforward=config.d_ff, output_size=1, dropout=config.dropout)
            # model = Model(config)
            run(config, model, Tranning_mode="normal_train")



   