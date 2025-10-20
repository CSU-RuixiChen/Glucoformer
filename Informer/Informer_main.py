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
from Informer.models.model import *


def get_Informer_parser():

    parser = argparse.ArgumentParser(description='Informer')

    # forecasting task
    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    parser.add_argument('--model_name', type=str, default='Informer', help='model name')
    parser.add_argument('--path_to_save_scaler', type=str, default="save_scaler/", help='device ids of multile gpus')
    parser.add_argument('--path_to_save_loss', type=str, default="../save_loss/", help='path to save loss')

    # dataset and dataloader
    parser.add_argument('--stride', type=int, default=1, help='the step ofstride window')
    parser.add_argument('--seq_len', type=int, default=360, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=180, help='start token length')
    parser.add_argument('--pred_len', type=int, default=90, help='prediction sequence length')
    parser.add_argument('--seg_len', type=int, default=0, help='segment length (L_seg)')
    parser.add_argument('--time_step', type=int, default=5, help='sensor_sampling')
    
    # model define
    parser.add_argument('--enc_in', type=int, default=3, help='encoder input size') # train_dataset.features_num
    parser.add_argument('--dec_in', type=int, default=3, help='decoder input size') # train_dataset.features_num
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=4, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=2, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dim_feedforward dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
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
    parser.add_argument('--batch_size', type=int, default=256, help='batch size of train input data')
    parser.add_argument('--train_epochs', type=int, default=50, help='train epochs')
    parser.add_argument('--patience', type=int, default=4, help='early stopping patience')
    parser.add_argument('--pre_lr', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--ft_lr', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--step_size', type=int, default=3, help='Period of learning rate decay')
    parser.add_argument('--pre_gamma', type=float, default=0.5, help='Multiplicative factor of learning rate decay')
    parser.add_argument('--ft_gamma', type=float, default=0.75, help='Multiplicative factor of learning rate decay')
    parser.add_argument('--tolerance', type=float, default=1, help='tolerance')
    parser.add_argument('--baseline', action='store_true', help='whether to use mean of past series as baseline for prediction', default=False)
    parser.add_argument('--save_pred', type=bool, default=True, help='whether to save the predicted future MTS')

    # GPU
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Device to use for computation: 'cpu' or 'cuda'")
    parser.add_argument('--best_pretrain_model', type=str, default=None, help='best_pretrain_model, eq: best_pretrain_pred(90min)_8.pth')
    parser.add_argument('--best_model', type=str, default=None, help='best_pretrain_model, eq: best_train_pred(60min)_12.pth')

    return parser


if __name__ == "__main__":
        
    parser = get_Informer_parser()
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
            if config.model_name == 'Informer':
                model = Informer(
                    enc_in=config.enc_in, dec_in=config.dec_in, c_out=config.c_out, out_len=pred_len,
                    factor=config.factor, d_model=config.d_model, n_heads=config.n_heads,
                    e_layers=config.e_layers, d_layers=config.d_layers, d_ff=config.d_ff, dropout=config.dropout
                )
            else:
                model = InformerStack(
                    enc_in=config.enc_in, dec_in=config.dec_in, c_out=config.c_out, out_len=pred_len,
                    factor=config.factor, d_model=config.d_model, n_heads=config.n_heads,
                    d_layers=config.d_layers, d_ff=config.d_ff, dropout=config.dropout
                )
            run(config, model, Tranning_mode="normal_train")