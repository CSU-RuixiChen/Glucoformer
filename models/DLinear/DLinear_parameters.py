import torch
import argparse

def get_DLinear_parser():

    parser = argparse.ArgumentParser(description='DLinear')

    # forecasting task
    parser.add_argument('--model_name', type=str, default='DLinear', help='model name')

    # DLinear
    parser.add_argument('--enc_in', type=int, default=3, help='encoder input size')
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')
    
    # optimization
    parser.add_argument('--batch_size', type=int, default=256, help='batch size of train input data')
    parser.add_argument('--train_epochs', type=int, default=50, help='train epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='optimizer learning rate')
    
    # ========== Model Loading Configuration ==========
    parser.add_argument('--pretrain_model_pth', type=str, default=None, help='path to pre-trained model')

    return parser