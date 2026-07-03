import torch
import argparse

def get_LSTM_parser():

    parser = argparse.ArgumentParser(description='LSTM')

    # forecasting task
    parser.add_argument('--model_name', type=str, default='LSTM', help='model name')

    # model define
    parser.add_argument('--data_dim', type=int, default=3, help='Number of dimensions of the MTS data (D)')# train_dataset.features_num
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout')

    # optimization
    parser.add_argument('--batch_size', type=int, default=256, help='batch size of train input data')
    parser.add_argument('--train_epochs', type=int, default=25, help='train epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='optimizer learning rate')

    # ========== Model Loading Configuration ==========
    parser.add_argument('--pretrain_model_pth', type=str, default=None, help='path to pre-trained model')

    return parser