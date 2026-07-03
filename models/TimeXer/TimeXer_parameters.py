import torch
import argparse

def get_TimeXer_parser():

    parser = argparse.ArgumentParser(description='TimeXer')

    # forecasting task
    parser.add_argument('--model_name', type=str, default='TimeXer', help='model name')
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')

    # model define
    parser.add_argument('--patch_len', type=int, default=15, help='segment length (L_seg)')
    parser.add_argument('--data_dim', type=int, default=3, help='Number of dimensions of the MTS data (D)')# train_dataset.features_num
    parser.add_argument('--enc_in', type=int, default=3, help='encoder input size') # train_dataset.features_num
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--d_model', type=int, default=256, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=512, help='dim_feedforward dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--factor', type=int, default=2, help='num of routers in Cross-Dimension Stage of TSA (c)')
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
    parser.add_argument('--train_epochs', type=int, default=25, help='train epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='optimizer learning rate')

    # ========== Model Loading Configuration ==========
    parser.add_argument('--pretrain_model_pth', type=str, default=None, help='path to pre-trained model')

    return parser
