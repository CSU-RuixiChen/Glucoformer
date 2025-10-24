import argparse
import torch
from typing import Optional


class GlucoformerOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Glucoformer')
        self.opt = None

    def _initial(self):
        # ========== Basic Task Configuration ==========
        self.parser.add_argument('--model_name', type=str, default='Glucoformer', help='model name')
        self.parser.add_argument('--path_to_save_scaler', type=str, default="save_scaler/", help='path to save scaler')
        self.parser.add_argument('--path_to_save_loss', type=str, default="../save_loss/", help='path to save loss')

        # ========== Dataset and DataLoader Configuration ==========
        self.parser.add_argument('--stride', type=int, default=1, help='the step of stride window')
        self.parser.add_argument('--seq_len', type=int, default=360, help='input sequence length')
        self.parser.add_argument('--label_len', type=int, default=180, help='start token length')
        self.parser.add_argument('--pred_len', type=int, default=90, help='prediction sequence length')
        self.parser.add_argument('--seg_len', type=int, default=15, help='segment length (L_seg)')
        self.parser.add_argument('--time_step', type=int, default=5, help='sensor sampling interval (minutes)')
        
        # ========== Model Architecture Configuration ==========
        self.parser.add_argument('--data_dim', type=int, default=3, help='Number of dimensions of the MTS data (D)')
        self.parser.add_argument('--c_out', type=int, default=1, help='output size')
        self.parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
        self.parser.add_argument('--n_heads', type=int, default=8, help='number of heads')
        self.parser.add_argument('--e_layers', type=int, default=3, help='number of encoder layers')
        self.parser.add_argument('--d_ff', type=int, default=512, help='dimension of feedforward network')
        self.parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
        self.parser.add_argument('--factor', type=int, default=1, help='number of routers in Cross-Dimension Stage of TSA (c)')
        self.parser.add_argument('--activation', type=str, default='gelu', help='activation function')

        # ========== Optimizer Configuration ==========
        self.parser.add_argument('--batch_size', type=int, default=256, help='batch size of training input data')
        self.parser.add_argument('--train_epochs', type=int, default=50, help='training epochs')
        self.parser.add_argument('--patience', type=int, default=4, help='early stopping patience')
        self.parser.add_argument('--pre_lr', type=float, default=0.0001, help='optimizer learning rate for pretraining')
        self.parser.add_argument('--ft_lr', type=float, default=0.001, help='optimizer learning rate for fine-tuning')
        self.parser.add_argument('--step_size', type=int, default=3, help='Period of learning rate decay')
        self.parser.add_argument('--pre_gamma', type=float, default=0.5, help='Multiplicative factor of learning rate decay for pretraining')
        self.parser.add_argument('--ft_gamma', type=float, default=0.75, help='Multiplicative factor of learning rate decay for fine-tuning')
        self.parser.add_argument('--tolerance', type=float, default=1.0, help='tolerance')
        self.parser.add_argument('--save_pred', type=bool, default=True, help='whether to save the predicted future MTS')

        # ========== Device Configuration ==========
        self.parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                                help="Device to use for computation: 'cpu' or 'cuda'")

        # ========== Model Loading Configuration ==========
        self.parser.add_argument('--best_pretrain_model', type=str, default=None, help='best pretrain model file name, e.g., best_pretrain_pred(60min)_7.pth')
        self.parser.add_argument('--best_model', type=str, default=None, help='best model file name, e.g., best_train_pred(90min)_12.pth')

        # ========== Optuna Optimization Configuration ==========
        self.parser.add_argument('--n_trials', type=int, default=50, help='number of optuna trials')
        self.parser.add_argument('--train_best', type=bool, default=True, help='whether to train the best model after optimization')

        # ========== Experiment Mode Configuration ==========
        self.parser.add_argument('--seed', type=int, default=2024, help='random seed')

    def parse(self, args=None):
        """Parse command line arguments"""
        self._initial()
        if args is None:
            self.opt = self.parser.parse_args()
        else:
            self.opt = self.parser.parse_args(args)
        return self.opt