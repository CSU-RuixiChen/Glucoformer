import os, shutil
import torch
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# save train or validation loss
def log_loss(path_to_save_loss: str, model_name: str, record: str, pred_length: int):
    file_name = f"save_loss_{model_name}_{pred_length}min.txt"
    path_to_file = path_to_save_loss + file_name
    os.makedirs(os.path.dirname(path_to_file), exist_ok=True)
    with open(path_to_file, "a", encoding="utf-8-sig") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{timestamp} "+record + "\n")
        f.close()

# Remove all files from previous executions and re-run the model.
def clean_directory(path_to_save_model):
    if os.path.exists(path_to_save_model):
        shutil.rmtree(path_to_save_model)
    os.mkdir(path_to_save_model)
    if os.path.exists('save_scaler'):
        shutil.rmtree('save_scaler')
    os.mkdir("save_scaler")


def generate_multi_feature_mask(decoder_input, nhead, device):
    batch_size, seq_len, _ = decoder_input.shape
    mask_shape = [batch_size*nhead , seq_len, seq_len]
    with torch.no_grad():
        mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)
    return mask

class EarlyStopping:
    def __init__(self, patience, pretrain:bool, verbose=False, delta=0):
        self.pretrain = pretrain
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.best_model = None  # 存储best_model

    def __call__(self, avg_validate_MSE_loss, model, path_to_save_model, forecast_window, epoch):
        score = -avg_validate_MSE_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model = self.save_model(avg_validate_MSE_loss, model, path_to_save_model, forecast_window, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model = self.save_model(avg_validate_MSE_loss, model, path_to_save_model, forecast_window, epoch)
            # self.save_prediton = save_predition(model_name, forecast_window, all_true_values, all_predicted_values, epoch)

            self.counter = 0

        return self.best_model  # 返回best_model

    def save_model(self, avg_validate_MSE_loss, model, path_to_save_model, forecast_window, epoch):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {avg_validate_MSE_loss:.6f}).  Saving model ...')
        if self.pretrain:
            best_model = f"best_pretrain_pred({forecast_window}min)_{epoch}.pth"
        else:
            best_model = f"best_train_pred({forecast_window}min)_{epoch}.pth"
        torch.save(model.state_dict(), path_to_save_model + best_model)
        self.val_loss_min = avg_validate_MSE_loss
        return best_model  # 返回模型文件名

    def get_best_model(self):
        return self.best_model  # 访问best_model


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj=='type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch-1) // 1))}
    elif args.lradj=='type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))
