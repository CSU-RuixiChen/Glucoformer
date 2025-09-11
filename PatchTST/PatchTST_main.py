import sys
import os
# 设置项目根目录路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
GLUCOFORMER_DIR = os.path.join(PROJECT_ROOT, "Glucoformer")
sys.path.insert(0, GLUCOFORMER_DIR)
sys.path.insert(0, os.path.dirname(GLUCOFORMER_DIR))
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from Glucose_Data.T1DMS_GlucoseDataset import T1DMS_GlucoseDataset
from Glucose_Data.OhioDataset import *
from Glucoformer.utils.train_eval import *
from Glucoformer.utils.tools import *
import random
import os
import argparse
from PatchTST.models.PatchTST import PatchTST


def get_PatchTST_parser():

    parser = argparse.ArgumentParser(description='PatchTST')

    # forecasting task
    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    parser.add_argument('--model_name', type=str, default='PatchTST1', help='model name')
    parser.add_argument('--path_to_save_scaler', type=str, default="save_scaler/", help='device ids of multile gpus')
    parser.add_argument('--path_to_save_loss', type=str, default="../save_loss/", help='path to save loss')

    # dataset and dataloader
    parser.add_argument('--stride', type=int, default= 1, help='the step of stride window')
    parser.add_argument('--seq_len', type=int, default=360, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=180, help='start token length')
    parser.add_argument('--pred_len', type=int, default=90, help='prediction sequence length')
    parser.add_argument('--time_step', type=int, default=5, help='sensor_sampling')

    # PatchTST
    parser.add_argument('--patch_len', type=int, default=15, help='patch length')
    parser.add_argument('--patch_stride', type=int, default=8, help='patch_stride')
    parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--revin', type=int, default=0, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
    parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
    parser.add_argument('--individual', type=int, default=1, help='individual head; True 1 False 0')
    
    # model define
    parser.add_argument('--enc_in', type=int, default=3, help='encoder input size') # train_dataset.features_num
    parser.add_argument('--dec_in', type=int, default=None, help='decoder input size') # train_dataset.features_num
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--d_model', type=int, default=256, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=512, help='dim_feedforward dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')

    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--use_norm', type=bool, default=False, help='use norm and denorm')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--attention_type', type=str, default="full", help='the attention type of transformer')
    # parser.add_argument('--embed', type=str, default='timeF',
    #                     help='time features encoding, options:[timeF, fixed, learned]')
    
    # optimization
    parser.add_argument('--pre_lr', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--ft_lr', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size of train input data')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--patience', type=int, default=4, help='early stopping patience')
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
    parser.add_argument('--save_pred', action='store_true', help='whether to save the predicted future MTS', default=True)

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
        
    parser = get_PatchTST_parser()
    config = parser.parse_args()

    Seeds = [2023, 2025]
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
            model = PatchTST(config)
            run(config, model, Tranning_mode="normal_train")
