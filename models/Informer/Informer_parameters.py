import torch
import argparse

def get_Informer_parser():

    parser = argparse.ArgumentParser(description='Informer')

    # forecasting task
    parser.add_argument('--model_name', type=str, default='Informer', help='model name')

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
    parser.add_argument('--train_epochs', type=int, default=25, help='train epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='optimizer learning rate')

    # ========== Model Loading Configuration ==========
    parser.add_argument('--pretrain_model_pth', type=str, default=None, help='path to pre-trained model')

    return parser