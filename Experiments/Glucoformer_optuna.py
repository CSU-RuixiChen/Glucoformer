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
    
    def __init__(self, base_config, n_trials=100, study_name="glucoformer_optimization"):
        """
        Initializes the Optuna optimizer.
        Args:
            n_trials: Optimization trial count
            study_name: Study name
        """
        self.base_config = base_config
        self.n_trials = n_trials
        self.study_name = study_name
        self.best_params = None
        self.best_value = None

        self.results_dir = f"optuna_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def suggest_hyperparameters(self, trial):
        
        d_model = trial.suggest_categorical('d_model', [128, 256])
        e_layers = trial.suggest_categorical('e_layers', [2, 3]) 
        d_ff = trial.suggest_categorical('d_ff', [512, 1024])
        factor = trial.suggest_int('factor', 1, 2) 
        batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
        pre_lr = trial.suggest_categorical('pre_lr', [1e-3, 5e-4, 1e-4]) 
        ft_lr = trial.suggest_categorical('ft_lr', [ 1e-3, 1e-4, 1e-5])
        dropout =  trial.suggest_categorical('dropout', [0.1, 0.2, 0.25])
        seg_len = trial.suggest_categorical('seg_len', [15, 20, 30])
        pre_gamma = trial.suggest_categorical('pre_gamma', [0.5, 0.6, 0.7]) 
        ft_gamma = trial.suggest_categorical('ft_gamma', [0.5, 0.65, 0.75]) 
        step_size = trial.suggest_int('step_size', 1, 3) 

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
        """Objective function for optimization."""
        
        # Get suggested hyperparameters
        suggested_params = self.suggest_hyperparameters(trial)
        
        # Create a new configuration object
        config = self.create_config_with_params(suggested_params)
        
        try:
            # Run training and validation
            val_loss = self.train_and_evaluate(config, trial.number)
            
            # Log the trial result
            self.log_trial_result(trial.number, suggested_params, val_loss)
            
            return val_loss
            
        except Exception as e:
            print(f"Trial {trial.number} failed with error: {str(e)}")
            # Return a large loss value to indicate failure
            return float('inf')
    
    def create_config_with_params(self, params):
        
        config = argparse.Namespace(**vars(self.base_config))
    
        for key, value in params.items():
            setattr(config, key, value)
            
        config.model_name = f"Glucoformer_trial"
        
        return config
    
    def train_and_evaluate(self, config, trial_number):
        
        fix_seed = 2024
        random.seed(fix_seed)
        torch.manual_seed(fix_seed)
        np.random.seed(fix_seed)
        
        time_step = config.time_step
        seq_len = int(config.seq_len / time_step)
        label_len = int(config.label_len / time_step)
        pred_len = int(config.pred_len / time_step)
        seg_len = int(config.seg_len / time_step)
        
        path_to_save_model = f"{self.results_dir}/trial_{trial_number}_model_{config.pred_len}min/"
        config.path_to_save_scaler = f"{self.results_dir}/trial_{trial_number}_scaler/"
        config.path_to_save_loss = f"{self.results_dir}/save_loss/"
        device = torch.device(config.device)
        
        if os.path.exists(path_to_save_model):
            shutil.rmtree(path_to_save_model)
        if os.path.exists(config.path_to_save_scaler):
            shutil.rmtree(config.path_to_save_scaler)
        os.makedirs(path_to_save_model, exist_ok=True)
        os.makedirs(config.path_to_save_scaler, exist_ok=True)
        
        try:
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
            
            train_dataset, validate_dataset, test_dataset, scalar = prepare_Ohio_data(
                data_dir="../Glucose_Data/OhioT1DM_processed_dataset",
                seq_length=seq_len, label_length=label_len, pred_length=pred_len,
                unimodal=False
            )
            
        except Exception as e:
            print(f"Data preparation failed: {str(e)}")
            raise e
        
        pretrain_dataloader = DataLoader(pretrain_dataset, config.batch_size, shuffle=True)
        prevalidate_dataloader = DataLoader(prevalidate_dataset, config.batch_size, shuffle=False)
        validate_dataloader = DataLoader(validate_dataset, config.batch_size, shuffle=False)
        
        model = Glucoformer(
            data_dim=config.data_dim, in_len=seq_len, out_len=pred_len,
            seg_len=seg_len, output_size=config.c_out,
            factor=config.factor, d_model=config.d_model, d_ff=config.d_ff,
            n_heads=config.n_heads, e_layers=config.e_layers,
            dropout=config.dropout
        ).to(device)
        
        best_pretrain_model = pretrain_model(
            config, model, pretrain_dataloader, prevalidate_dataloader,
            path_to_save_model, device
        )
        
        train_dataloader = DataLoader(train_dataset, config.batch_size, shuffle=True)
        best_model = Fine_Tuning_model(
            config, model, train_dataloader, validate_dataloader,
            scalar, path_to_save_model, best_pretrain_model, device
        )
        
        val_loss = self.evaluate_model(
            config, model, validate_dataloader, scalar,
            path_to_save_model, best_model, device
        )
        
        try:
            shutil.rmtree(path_to_save_model)
            shutil.rmtree(config.path_to_save_scaler)
        except:
            pass
        
        return val_loss
    
    def evaluate_model(self, config, model, dataloader, scalar, path_to_save_model, best_model, device):
        
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
                
                output_inverse = output * scalar.std[2] + scalar.mean[2]
                target_inverse = tgt * scalar.std[2] + scalar.mean[2]
                
                loss = criterion(output_inverse, target_inverse)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else float('inf')
    
    def log_trial_result(self, trial_number, params, val_loss):
        
        result = {
            'trial_number': trial_number,
            'parameters': params,
            'validation_loss': val_loss,
            'timestamp': datetime.now().isoformat()
        }
        
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
        
        print(f"Starting Optuna hyperparameter optimization with {self.n_trials} trials.")
        print(f"Results will be saved to: {self.results_dir}")
        
        # Create a study object
        db_path = os.path.join(self.results_dir, "my_optuna_results.db")
        study = optuna.create_study(
            direction='minimize',
            study_name=self.study_name,
            storage=f"sqlite:///{db_path}",  # Store results in an SQLite database
            sampler=optuna.samplers.TPESampler(seed=2024)
        )
        
        # Execute the optimization
        study.optimize(self.objective, n_trials=self.n_trials)
        
        # Save the best results
        self.best_params = study.best_params
        self.best_value = study.best_value
        
        # Save optimization results
        self.save_optimization_results(study)
        
        print("Optimization finished!")
        print(f"Best validation loss: {self.best_value:.6f}")
        print(f"Best parameters: {self.best_params}")
        
        return study
    
    def save_optimization_results(self, study):
        """Saves the optimization results."""
        
        # Save best parameters
        best_params_file = os.path.join(self.results_dir, 'best_parameters.json')
        with open(best_params_file, 'w') as f:
            json.dump({
                'best_params': self.best_params,
                'best_value': self.best_value,
                'n_trials': len(study.trials)
            }, f, indent=2)
        
        # Save detailed information for all trials
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
     
            # 1. Optimization History Plot
            plt.figure(figsize=(12, 8))
            optuna.visualization.matplotlib.plot_optimization_history(study)
            plt.title('Optimization History', fontsize=14, pad=20)
            plt.xlabel('Trial', fontsize=12)
            plt.ylabel('Objective Value', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'optimization_history.png'), dpi=600, bbox_inches='tight')
            plt.close()
            
            # 2. Parameter Importances Plot
            plt.figure(figsize=(12, 8))
            optuna.visualization.matplotlib.plot_param_importances(study)
            # plt.title('Parameter Importances', fontsize=14, pad=20)
            plt.xlabel('Importance', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'parameter_importances.png'), dpi=600, bbox_inches='tight')
            plt.close()
            
            # 3. Parallel Coordinate Plot
            ax = optuna.visualization.matplotlib.plot_parallel_coordinate(study)
            ax.figure.set_size_inches(15, 8)
            for label in ax.get_xticklabels():
                label.set_rotation(30)
                label.set_ha('right')

            for axes in ax.figure.axes:
                if axes.get_ylabel() == 'Objective Value':
                    axes.set_ylabel('MSE Loss', fontsize=12)

            xlabels = [label.get_text() for label in ax.get_xticklabels()]
            if len(xlabels) > 0 and xlabels[0] == 'Objective Value':
                xticklabels = ax.get_xticklabels()
                xticklabels[0].set_text('MSE Loss')
                ax.set_xticklabels([('MSE Loss' if t.get_text() == 'Objective Value' else t.get_text()) for t in xticklabels])

            # Set title and increase distance from plot
            ax.set_title(ax.get_title(), pad=25)
            # Adjust layout automatically
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, "parallel_coordinate.png"), dpi=600, bbox_inches="tight")
            plt.close()
            
            # 4. Convergence Plot
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
        """Trains the final model using the best parameters."""
        
        if self.best_params is None:
            raise ValueError("Optimization must be run before training the best model.")
        
        print("Training the final model with the best parameters...")
        
        # Create the best configuration
        best_config = self.create_config_with_params(self.best_params)
        best_config.model_name = "Glucoformer_best"
        best_config.train_epochs = self.base_config.train_epochs * 2  # Use more epochs to train the final model
        
        # Run the full training
        return self.run_full_training(best_config)
    
    def run_full_training(self, config):
        """Runs the full training process."""
        
        # Set random seed
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
        # Clean and create directories
        os.mkdir(f"{self.results_dir}/save_{config.model_name}_model_{config.pred_len}min")
        os.mkdir(f"{self.results_dir}/save_scaler")
        
        # Prepare data
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
        
        # Create data loaders
        pretrain_dataloader = DataLoader(pretrain_dataset, config.batch_size, shuffle=True)
        prevalidate_dataloader = DataLoader(prevalidate_dataset, config.batch_size, shuffle=False)
        train_dataloader = DataLoader(train_dataset, config.batch_size, shuffle=True)
        validate_dataloader = DataLoader(validate_dataset, config.batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, config.batch_size, shuffle=False)
        
        # Create model
        model = Glucoformer(
            data_dim=config.data_dim, in_len=seq_len, out_len=pred_len,
            seg_len=seg_len, output_size=config.c_out,
            factor=config.factor, d_model=config.d_model, d_ff=config.d_ff,
            n_heads=config.n_heads, e_layers=config.e_layers,
            dropout=config.dropout
        ).to(device)


        # Log configuration
        parameters_record = (f"-------------{config.model_name} pre-training starts-------------")
        print(parameters_record)
        log_loss(config.path_to_save_loss, config.model_name, parameters_record, config.pred_len)
        log_loss(config.path_to_save_loss, config.model_name, str(vars(config)), config.pred_len)
        
        # Pre-training
        best_pretrain_model = pretrain_model(
            config, model, pretrain_dataloader, prevalidate_dataloader,
            path_to_save_model, device
        )
        log_loss(config.path_to_save_loss, config.model_name, "Pre-training complete, fine-tuning begins", config.pred_len)
        log_loss(config.path_to_save_loss, config.model_name, best_pretrain_model, config.pred_len)
        
        # Fine-tuning
        print("-------------Pre-training complete, fine-tuning begins-------------")
        best_model = Fine_Tuning_model(
            config, model, train_dataloader, validate_dataloader,
            scalar, path_to_save_model, best_pretrain_model, device
        )
        log_loss(config.path_to_save_loss, config.model_name, best_model, config.pred_len)
        
        # Testing
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
    """Main Optuna optimization function."""
    
    # Create optimizer
    optimizer = OptunaOptimizer(
        base_config=config,
        n_trials=config.n_trials,
        study_name=f"glucoformer_opt_{config.pred_len}min"
    )
    
    # Execute optimization
    study = optimizer.optimize()
    
    # Train the best model
    if config.train_best:
        best_results = optimizer.train_best_model()
        print(f"Best model test results:")
        print(f"MSE: {best_results['test_rmse']** 2:.6f}")
        print(f"RMSE: {best_results['test_rmse']:.6f}")
        print(f"MAE: {best_results['test_mae']:.6f}")

    return study, optimizer


if __name__ == "__main__":
    
    config = GlucoformerOptions().parse()
    print('\nStarting Optuna hyperparameter optimization experiment...')
    print("\nOptions =================>")
    print(vars(config))

    config.n_trials = 50
    # config.train_epochs = 1

    study, optimizer = main_optuna(config)
