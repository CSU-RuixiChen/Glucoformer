import argparse
import torch
from typing import Optional

def get_Glucoformer_parser():

    parser = argparse.ArgumentParser(description='Glucoformer')

    # ========== Basic Task Configuration ==========
    parser.add_argument('--model_name', type=str, default='Glucoformer', help='model name')
    
    # ========== Model Architecture Configuration ==========
    parser.add_argument('--data_dim', type=int, default=3, help='Number of dimensions of the MTS data (D)')
    parser.add_argument('--seg_len', type=int, default=30, help='segment length (L_seg)')
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='number of heads')

    parser.add_argument('--e_layers', type=int, default=3, help='number of encoder layers')
    parser.add_argument('--d_ff', type=int, default=256, help='dimension of feedforward network')

    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
    parser.add_argument('--factor', type=int, default=1, help='number of routers in Cross-Dimension Stage of TSA (c)')
    parser.add_argument('--activation', type=str, default='gelu', help='activation function')

    parser.add_argument('--use_sse', type=bool, default=True, help='')
    parser.add_argument('--use_etsa', type=bool, default=True, help='')
    parser.add_argument('--use_decoder_self_attn', type=bool, default=False, help='')

    # ========== Optimizer Configuration ==========
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of training input data')
    parser.add_argument('--train_epochs', type=int, default=25, help='training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='optimizer learning rate for fine-tuning')

    # parser.add_argument('--pretrain_model_pth', type=str, 
    #                     default='results/T1DMS_Glucoformer_pretrain0_bs512_lr1.5e-04_dm128_el3_2026-06-11_110335/checkpoints/Glucoformer_seed0.pth', 
    #                     help='path to pre-trained model')

    # parser.add_argument('--pretrain_model_pth', type=str, 
    #                     default='results/T1DMS_Glucoformer2_pretrain0_bs512_lr1.5e-04_dm128_el3_2026-05-29_122625/checkpoints/Glucoformer2_seed0.pth', 
    #                     help='path to pre-trained model')
    # parser.add_argument('--pretrain_model_pth', type=str, 
    #                     default='results/T1DMS_Glucoformer2_pretrain0_bs512_lr1.5e-04_dm128_el3_2026-05-29_134614/checkpoints/Glucoformer2_seed0.pth', 
    #                     help='path to pre-trained model')# Ins and CHO = -1


    # parser.add_argument('--batch_size', type=int, default=512, help='batch size of training input data')
    # parser.add_argument('--train_epochs', type=int, default=25, help='training epochs')
    # parser.add_argument('--lr', type=float, default=0.00015, help='optimizer learning rate for fine-tuning')
    parser.add_argument('--pretrain_model_pth', type=str, default=None, help='path to pre-trained model')


    return parser