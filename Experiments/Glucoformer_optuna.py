import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
GLUCOFORMER_DIR = os.path.join(PROJECT_ROOT, "Glucoformer")
sys.path.insert(0, GLUCOFORMER_DIR)
sys.path.insert(0, os.path.dirname(GLUCOFORMER_DIR))
from Glucose_Data.T1DMS_GlucoseDataset import T1DMS_GlucoseDataset
from Glucose_Data.OhioDataset import *
from Glucoformer.Glucoformer_models.Glucoformer import *
from Glucoformer.utils.train_eval import *
from Glucoformer.utils.Glucoformer_options import GlucoformerOptions
from Glucoformer.utils.tools import *
import random
import argparse
import optuna
import torch
import json
import numpy as np
from datetime import datetime


class OptunaOptimizer:
    """Optuna超参数优化器类"""
    
    def __init__(self, base_config, n_trials=100, study_name="glucoformer_optimization"):
        """
        初始化优化器
        
        Args:
            base_config: 基础配置对象
            n_trials: 优化试验次数
            study_name: 研究名称
        """
        self.base_config = base_config
        self.n_trials = n_trials
        self.study_name = study_name
        self.best_params = None
        self.best_value = None
        
        # 创建结果保存目录
        self.results_dir = f"optuna_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def suggest_hyperparameters(self, trial):
        """定义超参数搜索空间"""
        
        # 模型结构741相关参数
        d_model = trial.suggest_categorical('d_model', [128, 256])
        e_layers = trial.suggest_categorical('e_layers', [2, 3]) # 正确：在2和3之间选择
        d_ff = trial.suggest_categorical('d_ff', [512, 1024])
        factor = trial.suggest_int('factor', 1, 2)  # 在1和2之间选择
            
        # 训练相关参数
        batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
        pre_lr = trial.suggest_categorical('pre_lr', [1e-3, 5e-4, 1e-4]) 
        ft_lr = trial.suggest_categorical('ft_lr', [ 1e-3, 1e-4, 1e-5])
        dropout =  trial.suggest_categorical('dropout', [0.1, 0.2, 0.25])
        # 序列长度相关参数
        seg_len = trial.suggest_categorical('seg_len', [15, 20, 30])
        
        # 优化器相关参数
        pre_gamma = trial.suggest_categorical('pre_gamma', [0.5, 0.6, 0.7])  # 修正：只有low和high
        ft_gamma = trial.suggest_categorical('ft_gamma', [0.5, 0.65, 0.75])      # 修正：只有low和high
        step_size = trial.suggest_int('step_size', 1, 3)         # 修正：只有low和high

        return {
            'd_model': d_model,
            'e_layers': e_layers,
            'd_ff': d_ff,
            'factor': factor,
            'batch_size': batch_size,
            'pre_lr': pre_lr,
            'ft_lr': ft_lr,
            'dropout': dropout,
            'seg_len': seg_len,
            'pre_gamma': pre_gamma,
            'ft_gamma': ft_gamma,
            'step_size': step_size
        }
    
    def objective(self, trial):
        """优化目标函数"""
        
        # 获取建议的超参数
        suggested_params = self.suggest_hyperparameters(trial)
        
        # 创建新的配置对象
        config = self.create_config_with_params(suggested_params)
        
        try:
            # 运行训练和验证
            val_loss = self.train_and_evaluate(config, trial.number)
            
            # 记录试验结果
            self.log_trial_result(trial.number, suggested_params, val_loss)
            
            return val_loss
            
        except Exception as e:
            print(f"Trial {trial.number} failed with error: {str(e)}")
            # 返回一个很大的损失值表示失败
            return float('inf')
    
    def create_config_with_params(self, params):
        """根据建议的参数创建配置对象"""
        
        # 复制基础配置
        config = argparse.Namespace(**vars(self.base_config))
        
        # 更新参数
        for key, value in params.items():
            setattr(config, key, value)
            
        # 更新模型保存路径以避免冲突
        config.model_name = f"Glucoformer_trial"
        
        return config
    
    def train_and_evaluate(self, config, trial_number):
        """训练和评估模型"""
        
        # 设置随机种子
        fix_seed = 2024
        random.seed(fix_seed)
        torch.manual_seed(fix_seed)
        np.random.seed(fix_seed)
        
        time_step = config.time_step
        seq_len = int(config.seq_len / time_step)
        label_len = int(config.label_len / time_step)
        pred_len = int(config.pred_len / time_step)
        seg_len = int(config.seg_len / time_step)
        
        # 创建试验特定的保存路径
        path_to_save_model = f"{self.results_dir}/trial_{trial_number}_model_{config.pred_len}min/"
        config.path_to_save_scaler = f"{self.results_dir}/trial_{trial_number}_scaler/"
        config.path_to_save_loss = f"{self.results_dir}/save_loss/"
        device = torch.device(config.device)
        
        # 清理目录
        if os.path.exists(path_to_save_model):
            shutil.rmtree(path_to_save_model)
        if os.path.exists(config.path_to_save_scaler):
            shutil.rmtree(config.path_to_save_scaler)
        os.makedirs(path_to_save_model, exist_ok=True)
        os.makedirs(config.path_to_save_scaler, exist_ok=True)
        
        # 准备数据
        try:
            # 预训练数据
            pretrain_dataset = T1DMS_GlucoseDataset(
                sim_result_mat_file_name='sim_results_train_dataset.mat',
                sim_data_mat_file_name='sim_data_train_dataset.mat',
                root_dir='../Glucose_Data/T1DMS_GlucoseDataset',
                train_dataset=True,
                seq_length=seq_len, label_length=label_len, pred_length=pred_len,
                stride=config.stride, sensor_sampling=time_step,
                path_to_save_scaler=config.path_to_save_scaler
            )
            
            prevalidate_dataset = T1DMS_GlucoseDataset(
                sim_result_mat_file_name='sim_results_validate_dataset.mat',
                sim_data_mat_file_name='sim_data_validate_dataset.mat',
                root_dir='../Glucose_Data/T1DMS_GlucoseDataset',
                train_dataset=False,
                seq_length=seq_len, label_length=label_len, pred_length=pred_len,
                stride=config.stride, sensor_sampling=time_step,
                path_to_save_scaler=config.path_to_save_scaler
            )
            
            # 微调数据
            train_dataset, validate_dataset, test_dataset, scalar = prepare_Ohio_data(
                data_dir="../Glucose_Data/OhioT1DM_processed_dataset",
                seq_length=seq_len, label_length=label_len, pred_length=pred_len,
                unimodal=False
            )
            
        except Exception as e:
            print(f"Data preparation failed: {str(e)}")
            raise e
        
        # 创建数据加载器
        pretrain_dataloader = DataLoader(pretrain_dataset, config.batch_size, shuffle=True)
        prevalidate_dataloader = DataLoader(prevalidate_dataset, config.batch_size, shuffle=False)
        validate_dataloader = DataLoader(validate_dataset, config.batch_size, shuffle=False)
        
        # 创建模型
        model = Glucoformer(
            data_dim=config.data_dim, in_len=seq_len, out_len=pred_len,
            seg_len=seg_len, output_size=config.c_out,
            factor=config.factor, d_model=config.d_model, d_ff=config.d_ff,
            n_heads=config.n_heads, e_layers=config.e_layers,
            dropout=config.dropout
        ).to(device)
        
        # 预训练
        best_pretrain_model = pretrain_model(
            config, model, pretrain_dataloader, prevalidate_dataloader,
            path_to_save_model, device
        )
        
        # 微调
        train_dataloader = DataLoader(train_dataset, config.batch_size, shuffle=True)
        best_model = Fine_Tuning_model(
            config, model, train_dataloader, validate_dataloader,
            scalar, path_to_save_model, best_pretrain_model, device
        )
        
        # 计算验证损失
        val_loss = self.evaluate_model(
            config, model, validate_dataloader, scalar,
            path_to_save_model, best_model, device
        )
        
        # 清理模型文件以节省空间
        try:
            shutil.rmtree(path_to_save_model)
            shutil.rmtree(config.path_to_save_scaler)
        except:
            pass
        
        return val_loss
    
    def evaluate_model(self, config, model, dataloader, scalar, path_to_save_model, best_model, device):
        """评估模型性能"""
        
        model.load_state_dict(torch.load(path_to_save_model + best_model))
        criterion = torch.nn.MSELoss()
        
        model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for encoder_input, encoder_input_mark, decoder_input, decoder_input_mark, tgt, _ in dataloader:
                encoder_input = encoder_input.to(device)
                encoder_input_mark = encoder_input_mark.to(device)
                decoder_input = decoder_input.to(device)
                decoder_input_mark = decoder_input_mark.to(device)
                tgt = tgt.to(device)
                
                output = model(encoder_input, encoder_input_mark, decoder_input, decoder_input_mark)
                
                # 反归一化
                output_inverse = output * scalar.std[2] + scalar.mean[2]
                target_inverse = tgt * scalar.std[2] + scalar.mean[2]
                
                loss = criterion(output_inverse, target_inverse)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else float('inf')
    
    def log_trial_result(self, trial_number, params, val_loss):
        """记录试验结果"""
        
        result = {
            'trial_number': trial_number,
            'parameters': params,
            'validation_loss': val_loss,
            'timestamp': datetime.now().isoformat()
        }
        
        # 保存到JSON文件
        results_file = os.path.join(self.results_dir, 'trial_results.json')
        
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
        else:
            results = []
        
        results.append(result)
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    def optimize(self):
        """执行超参数优化"""
        
        print(f"开始Optuna超参数优化，试验次数: {self.n_trials}")
        print(f"结果将保存到: {self.results_dir}")
        
        # 创建研究对象
        db_path = os.path.join(self.results_dir, "my_optuna_results.db")
        study = optuna.create_study(
            direction='minimize',
            study_name=self.study_name,
            storage=f"sqlite:///{db_path}",  # 存储到SQLite数据库
            sampler=optuna.samplers.TPESampler(seed=2024)
        )
        
        # 执行优化
        study.optimize(self.objective, n_trials=self.n_trials)
        
        # 保存最佳结果
        self.best_params = study.best_params
        self.best_value = study.best_value
        
        # 保存优化结果
        self.save_optimization_results(study)
        
        print(f"优化完成！")
        print(f"最佳验证损失: {self.best_value:.6f}")
        print(f"最佳参数: {self.best_params}")
        
        return study
    
    def save_optimization_results(self, study):
        """保存优化结果"""
        
        # 保存最佳参数
        best_params_file = os.path.join(self.results_dir, 'best_parameters.json')
        with open(best_params_file, 'w') as f:
            json.dump({
                'best_params': self.best_params,
                'best_value': self.best_value,
                'n_trials': len(study.trials)
            }, f, indent=2)
        
        # 保存所有试验的详细信息
        trials_data = []
        for trial in study.trials:
            trial_data = {
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': trial.state.name,
                'datetime_start': trial.datetime_start.isoformat() if trial.datetime_start else None,
                'datetime_complete': trial.datetime_complete.isoformat() if trial.datetime_complete else None
            }
            trials_data.append(trial_data)
        
        trials_file = os.path.join(self.results_dir, 'all_trials.json')
        with open(trials_file, 'w') as f:
            json.dump(trials_data, f, indent=2)
        
        try:
            import matplotlib.pyplot as plt
     
            # 1. 优化历史图
            plt.figure(figsize=(12, 8))
            optuna.visualization.matplotlib.plot_optimization_history(study)
            plt.title('Optimization History', fontsize=14, pad=20)
            plt.xlabel('Trial', fontsize=12)
            plt.ylabel('Objective Value', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'optimization_history.png'), dpi=600, bbox_inches='tight')
            plt.close()
            
            # 2. 参数重要性图
            plt.figure(figsize=(12, 8))
            optuna.visualization.matplotlib.plot_param_importances(study)
            # plt.title('Parameter Importances', fontsize=14, pad=20)
            plt.xlabel('Importance', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'parameter_importances.png'), dpi=600, bbox_inches='tight')
            plt.close()
            
            # 3. 平行坐标图
            ax = optuna.visualization.matplotlib.plot_parallel_coordinate(study)
            ax.figure.set_size_inches(15, 8)
            for label in ax.get_xticklabels():
                label.set_rotation(30)
                label.set_ha('right')

            for axes in ax.figure.axes:
                if axes.get_ylabel() == 'Objective Value':
                    axes.set_ylabel('MSE_Loss', fontsize=12)

            xlabels = [label.get_text() for label in ax.get_xticklabels()]
            if len(xlabels) > 0 and xlabels[0] == 'Objective Value':
                xticklabels = ax.get_xticklabels()
                xticklabels[0].set_text('MSE_Loss')
                ax.set_xticklabels([('MSE_Loss' if t.get_text() == 'Objective Value' else t.get_text()) for t in xticklabels])

            # 设置标题并加大与图的距离
            ax.set_title(ax.get_title(), pad=25)
            # 自动调整布局
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, "parallel_coordinate.png"), dpi=600, bbox_inches="tight")
            plt.close()
            
            # 4. 收敛图
            plt.figure(figsize=(12, 8))
            values = [trial.value for trial in study.trials if trial.value is not None]
            best_values = []
            best_so_far = float('inf')
            for value in values:
                if value < best_so_far:
                    best_so_far = value
                best_values.append(best_so_far)
            
            plt.plot(best_values, linewidth=2.5, color='royalblue')
            plt.scatter(range(len(best_values)), best_values, 
                    c=best_values, cmap='viridis', 
                    s=50, alpha=0.7, edgecolors='k')
            
            plt.xlabel('Trial', fontsize=12)
            plt.ylabel('Best Value So Far', fontsize=12)
            plt.title('Convergence Plot', fontsize=14, pad=20)
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'convergence_plot.png'), dpi=600, bbox_inches='tight')
            plt.close()
            
            print(f"Optimization plots saved separately in: {self.results_dir}")
            
        except ImportError:
            print("matplotlib not available, skipping plots")
        except Exception as e:
            print(f"Error generating plots: {e}")

        
    def train_best_model(self):
        """使用最佳参数训练最终模型"""
        
        if self.best_params is None:
            raise ValueError("需要先运行优化才能训练最佳模型")
        
        print("使用最佳参数训练最终模型...")
        
        # 创建最佳配置
        best_config = self.create_config_with_params(self.best_params)
        best_config.model_name = "Glucoformer_best"
        best_config.train_epochs = self.base_config.train_epochs * 2  # 使用更多epoch训练最终模型
        
        # 运行完整训练
        return self.run_full_training(best_config)
    
    def run_full_training(self, config):
        """运行完整的训练流程"""
        
        # 设置随机种子
        fix_seed = config.seed
        random.seed(fix_seed)
        torch.manual_seed(fix_seed)
        np.random.seed(fix_seed)
        
        time_step = config.time_step
        seq_len = int(config.seq_len / time_step)
        label_len = int(config.label_len / time_step)
        pred_len = int(config.pred_len / time_step)
        seg_len = int(config.seg_len / time_step)
        
        path_to_save_model = f"{self.results_dir}/save_{config.model_name}_model_{config.pred_len}min/"
        config.path_to_save_scaler = f"{self.results_dir}/save_scaler/"
        config.path_to_save_loss = f"{self.results_dir}/save_loss/"
        device = torch.device(config.device)
        # 清理并创建目录
        os.mkdir(f"{self.results_dir}/save_{config.model_name}_model_{config.pred_len}min")
        os.mkdir(f"{self.results_dir}/save_scaler")
        
        # 准备数据
        pretrain_dataset = T1DMS_GlucoseDataset(
            sim_result_mat_file_name='sim_results_train_dataset.mat',
            sim_data_mat_file_name='sim_data_train_dataset.mat',
            root_dir='../../Glucose_Data/T1DMS_GlucoseDataset',
            train_dataset=True,
            seq_length=seq_len, label_length=label_len, pred_length=pred_len,
            stride=config.stride, sensor_sampling=time_step,
            path_to_save_scaler=config.path_to_save_scaler
        )
        
        prevalidate_dataset = T1DMS_GlucoseDataset(
            sim_result_mat_file_name='sim_results_validate_dataset.mat',
            sim_data_mat_file_name='sim_data_validate_dataset.mat',
            root_dir='../../Glucose_Data/T1DMS_GlucoseDataset',
            train_dataset=False,
            seq_length=seq_len, label_length=label_len, pred_length=pred_len,
            stride=config.stride, sensor_sampling=time_step,
            path_to_save_scaler=config.path_to_save_scaler
        )
        
        train_dataset, validate_dataset, test_dataset, scalar = prepare_Ohio_data(
            data_dir="../../Glucose_Data/OhioT1DM_processed_dataset",
            seq_length=seq_len, label_length=label_len, pred_length=pred_len,
            unimodal=False
        )
        
        # 创建数据加载器
        pretrain_dataloader = DataLoader(pretrain_dataset, config.batch_size, shuffle=True)
        prevalidate_dataloader = DataLoader(prevalidate_dataset, config.batch_size, shuffle=False)
        train_dataloader = DataLoader(train_dataset, config.batch_size, shuffle=True)
        validate_dataloader = DataLoader(validate_dataset, config.batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, config.batch_size, shuffle=False)
        
        # 创建模型
        model = Glucoformer(
            data_dim=config.data_dim, in_len=seq_len, out_len=pred_len,
            seg_len=seg_len, output_size=config.c_out,
            factor=config.factor, d_model=config.d_model, d_ff=config.d_ff,
            n_heads=config.n_heads, e_layers=config.e_layers,
            dropout=config.dropout
        ).to(device)


        # 记录配置
        parameters_record = (f"-------------{config.model_name} pre-training starts-------------")
        print(parameters_record)
        log_loss(config.path_to_save_loss, config.model_name, parameters_record, config.pred_len)
        log_loss(config.path_to_save_loss, config.model_name, str(vars(config)), config.pred_len)
        
        # 预训练
        best_pretrain_model = pretrain_model(
            config, model, pretrain_dataloader, prevalidate_dataloader,
            path_to_save_model, device
        )
        log_loss(config.path_to_save_loss, config.model_name, "Pre-training complete, fine-tuning begins", config.pred_len)
        log_loss(config.path_to_save_loss, config.model_name, best_pretrain_model, config.pred_len)
        
        # 微调
        print("-------------Pre-training complete, fine-tuning begins-------------")
        best_model = Fine_Tuning_model(
            config, model, train_dataloader, validate_dataloader,
            scalar, path_to_save_model, best_pretrain_model, device
        )
        log_loss(config.path_to_save_loss, config.model_name, best_model, config.pred_len)
        
        # 测试
        print("-------------Fine-tuning complete, testing begins-------------")
        test_rmse, test_mae = prediction(
            config, model, test_dataloader, scalar,
            path_to_save_model, best_model, device
        )
        
        return {
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'best_model': best_model,
            'config': config
        }


def main_optuna(config):
    """主要的Optuna优化函数"""
    
    # 创建优化器
    optimizer = OptunaOptimizer(
        base_config=config,
        n_trials=config.n_trials,
        study_name=f"glucoformer_opt_{config.pred_len}min"
    )
    
    # 执行优化
    study = optimizer.optimize()
    
    # 训练最佳模型
    if config.train_best:
        best_results = optimizer.train_best_model()
        print(f"最佳模型测试结果:")
        print(f"MSE: {best_results['test_rmse']** 2:.6f}")
        print(f"RMSE: {best_results['test_rmse']:.6f}")
        print(f"MAE: {best_results['test_mae']:.6f}")

    return study, optimizer


if __name__ == "__main__":
    
    config = GlucoformerOptions().parse()
    print('\nOptuna超参数优化实验开始...')
    print("\nOptions =================>")
    print(vars(config))

    config.n_trials = 50
    # config.train_epochs = 1

    study, optimizer = main_optuna(config)
