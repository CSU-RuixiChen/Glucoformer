import os
import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def merge_options(global_opt, model_parser, args=None):
    if getattr(global_opt, "skip_merge", False):
        return global_opt
    # parse model
    if args is None:
        import sys
        if sys.argv[0].endswith('ipykernel_launcher.py') or 'jupyter' in sys.argv[0]:
            model_opt, _ = model_parser.parse_known_args([])
        else:
            model_opt, _ = model_parser.parse_known_args()
    else:
        model_opt, _ = model_parser.parse_known_args(args)

    merged = vars(global_opt).copy()
    merged.update(vars(model_opt))

    return argparse.Namespace(**merged)


def setup_folders(LOG_DIR, config, clear=False):
    """
    Create experiment data save folders.
    If clear=True, existing folders will be deleted and recreated.
    If clear=False, folders will only be created if they do not exist (no deletion).
    """
    import shutil
    base = LOG_DIR
    rel_paths = [config.model_save_path, config.loss_record_path, 
                 config.eval_result_path, config.analysis_result_path]
    paths = [base + rel for rel in rel_paths]
    for path in paths:
        if clear and os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)


def set_seed(seed=2024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    print(f"\n{'='*10} Random seed set to {seed} {'='*10}\n")


def plot_loss_curve(loss_record_csv_path, fig_save_path):
    df = pd.read_csv(loss_record_csv_path)
    epoch_list = df['epoch'].tolist()
    train_rmse_list = df['train_RMSE'].tolist()
    val_rmse_list = df['val_RMSE'].tolist()

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('RMSE')

    line1, = ax1.plot(epoch_list, val_rmse_list, color='blue', linewidth=1, label='val_RMSE')
    line2, = ax1.plot(epoch_list, train_rmse_list, color='red', linewidth=1, label='train_RMSE')

    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best')
    fig.tight_layout()
    plt.savefig(fig_save_path, dpi=300)
    plt.close()

