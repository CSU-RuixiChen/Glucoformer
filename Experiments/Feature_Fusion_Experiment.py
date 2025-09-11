import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
GLUCOFORMER_DIR = os.path.join(PROJECT_ROOT, "Glucoformer")
sys.path.insert(0, GLUCOFORMER_DIR)
sys.path.insert(0, os.path.dirname(GLUCOFORMER_DIR))
from Glucoformer.utils.train_eval import *
from Glucoformer.utils.tools import *
import random
import os

from Glucoformer.utils.Glucoformer_options import GlucoformerOptions
from Glucoformer.Glucoformer_models.Glucoformer import *

config = GlucoformerOptions().parse()

result_dir = f"feature_fusion_{config.model_name}_results"
result_csv = os.path.join(result_dir, "raw_results.csv")

if os.path.exists(result_dir):
    shutil.rmtree(result_dir)
os.makedirs(result_dir, exist_ok=True)

Seeds = [2023, 2024, 2025]
Experiment_modes = ['ggg', 'cgg', 'gig']
prediction_task = [30, 60, 90]

# 每次实验结果直接写入CSV
with open(result_csv, "w") as f:
    f.write("mode,ph,seed,RMSE,MAE\n")

for Experiment_mode in Experiment_modes:
    for pred_len_raw in prediction_task:
        for seed in Seeds:
            config.seed = seed
            random.seed(config.seed)
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)
            config.pred_len = pred_len_raw
            config.label_len = 2 * pred_len_raw
            config.seq_len = 4 * pred_len_raw
            seq_len = int(config.seq_len/config.time_step)
            label_len = int(config.label_len/config.time_step)
            pred_len = int(config.pred_len/config.time_step)
            seg_len = int(config.seg_len/config.time_step)
            config.enc_in = config.data_dim
            model = Glucoformer(
                data_dim=config.data_dim, in_len=seq_len, out_len=pred_len, seg_len=seg_len, output_size=config.c_out,
                factor=config.factor, d_model=config.d_model, d_ff=config.d_ff, n_heads=config.n_heads,
                e_layers=config.e_layers, dropout=config.dropout)
            RMSE, MAE = run(config, model, Tranning_mode="pretrain", Experiment_mode=Experiment_mode)
            # 追加写入结果，顺序为mode,ph,seed
            with open(result_csv, "a") as f:
                f.write(f"{Experiment_mode},{pred_len_raw},{seed},{RMSE},{MAE}\n")

# 统计分析
df = pd.read_csv(result_csv)
rows = []
for mode in Experiment_modes:
    for ph in prediction_task:
        sub = df[(df['mode'] == mode) & (df['ph'] == ph)]
        rmse_mean, rmse_std = sub['RMSE'].mean(), sub['RMSE'].std()
        mae_mean, mae_std = sub['MAE'].mean(), sub['MAE'].std()
        rows.append({
            'Mode': mode,
            'PH': ph,
            'RMSE': f"{rmse_mean:.2f}±{rmse_std:.2f}",
            'MAE': f"{mae_mean:.2f}±{mae_std:.2f}"
        })

df_stat = pd.DataFrame(rows)
print(df_stat)
df_stat.to_csv(os.path.join(result_dir, "feature_fusion_results.csv"), index=False, encoding='utf-8')