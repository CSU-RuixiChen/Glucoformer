import torch
import argparse

def get_Crossformer_parser():
   
    parser = argparse.ArgumentParser(description='Crossformer')

    # forecasting task
    parser.add_argument('--model_name', type=str, default='Crossformer', help='model name')

    # model define
    parser.add_argument('--data_dim', type=int, default=3, help='Number of dimensions of the MTS data (D)')# train_dataset.features_num
    parser.add_argument('--seg_len', type=int, default=6, help='segment length (L_seg)')
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--d_model', type=int, default=256, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--d_ff', type=int, default=512, help='dim_feedforward dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
    parser.add_argument('--win_size', type=int, default=2, help='window size for segment merge')
    parser.add_argument('--factor', type=int, default=1, help='num of routers in Cross-Dimension Stage of TSA (c)')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')

    # optimization
    parser.add_argument('--batch_size', type=int, default=256, help='batch size of train input data')
    parser.add_argument('--train_epochs', type=int, default=25, help='train epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='optimizer learning rate')

    # ========== Model Loading Configuration ==========
    parser.add_argument('--pretrain_model_pth', type=str, default=None, help='path to pre-trained model')

    return parser