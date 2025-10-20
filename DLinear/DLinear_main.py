import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
GLUCOFORMER_DIR = os.path.join(PROJECT_ROOT, "Glucoformer")
sys.path.insert(0, GLUCOFORMER_DIR)
sys.path.insert(0, os.path.dirname(GLUCOFORMER_DIR))
from Glucoformer.utils.train_eval import *
import random
import argparse
from DLinear.DLinear_model import DLinear

def get_DLinear_parser():

    parser = argparse.ArgumentParser(description='DLinear')

    # forecasting task
    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    parser.add_argument('--model_name', type=str, default='DLinear', help='model name')
    parser.add_argument('--path_to_save_scaler', type=str, default="save_scaler/", help='device ids of multile gpus')
    parser.add_argument('--path_to_save_loss', type=str, default="../save_loss/", help='path to save loss')

    # dataset and dataloader
    parser.add_argument('--stride', type=int, default=1, help='the step ofstride window')
    parser.add_argument('--seq_len', type=int, default=360, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=180, help='start token length')
    parser.add_argument('--pred_len', type=int, default=90, help='prediction sequence length')
    parser.add_argument('--seg_len', type=int, default=0, help='segment length (L_seg)')
    parser.add_argument('--time_step', type=int, default=5, help='sensor_sampling')
    
    # DLinear
    parser.add_argument('--enc_in', type=int, default=3, help='encoder input size')
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--individual', type=int, default=1, help='individual head; True 1 False 0')
    
    # optimization
    parser.add_argument('--batch_size', type=int, default=256, help='batch size of train input data')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--patience', type=int, default=4, help='early stopping patience')
    parser.add_argument('--pre_lr', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--ft_lr', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--step_size', type=int, default=3, help='Period of learning rate decay')
    parser.add_argument('--pre_gamma', type=float, default=0.5, help='Multiplicative factor of learning rate decay')
    parser.add_argument('--ft_gamma', type=float, default=0.75, help='Multiplicative factor of learning rate decay')
    parser.add_argument('--tolerance', type=float, default=1, help='tolerance')
    # parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
    # parser.add_argument('--des', type=str, default='test', help='exp description')
    # parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    # parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    # parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--baseline', action='store_true', help='whether to use mean of past series as baseline for prediction', default=False)
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--save_pred', type=bool, default=True, help='whether to save the predicted future MTS')

    # GPU
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Device to use for computation: 'cpu' or 'cuda'")
    # parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    # parser.add_argument('--gpu', type=int, default=0, help='gpu')
    # parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    # parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--best_pretrain_model', type=str, default=None, help='best_pretrain_model, eq: best_pretrain_pred(90min)_8.pth')
    parser.add_argument('--best_model', type=str, default=None, help='best_pretrain_model, eq: best_train_pred(60min)_12.pth')

    return parser


if __name__ == "__main__":
        
    parser = get_DLinear_parser()
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
            model = DLinear(config)
            run(config, model, Tranning_mode="normal_train")

     



   