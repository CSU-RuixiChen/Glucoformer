import warnings, os
import json
warnings.filterwarnings("ignore")
import torch
import math
import datetime
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from data_provider.T1DMS_Dataset import DataLoader, prepare_T1DMS_data
from data_provider.OhioT1DM_Dataset import prepare_Ohio_data
from data_provider.DiaTrend_Dataset import prepare_DiaTrend_data, prepare_DiaTrend_zeroshot_data
from exp.loss import MultiHorizonMSELoss
from utils.tools import *
import glob
import os
import json
import glob
import datetime
import pandas as pd
from torch.utils.data import DataLoader


def main(config, main_path=os.getcwd()):
    
    for seed in range(0, config.repeat_times):
        config.seed = seed
        set_seed(config.seed)
        
        config, model, device = set_model_parameters(config)

        if "T1DMS" in config.dataset:
            config.pretrain_model_pth = None

        if seed == 0:
            if config.experiment_dir is None:
                time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
                folder_id = '{}_{}_pretrain{}_bs{}_lr{:.1e}_dm{}_el{}_{}'.format(
                            config.dataset,
                            config.model_name,               
                            1 if (config.pretrain_model_pth is not None and config.pretrain_model_pth != 'None') else 0,  
                            config.batch_size,                   
                            config.lr,  
                            getattr(config, 'd_model', 0), 
                            getattr(config, 'e_layers', 0),
                            time_stamp
                        )
                config.experiment_dir = os.path.join("results", folder_id)
                LOG_DIR =  os.path.join(main_path, config.experiment_dir)
                setup_folders(LOG_DIR, config)

                with open(os.path.join(LOG_DIR, f"config_{time_stamp}.json"), 'w') as f:
                    json.dump(vars(config), f, indent=4)   
            else:
                LOG_DIR = os.path.join(main_path, config.experiment_dir)
                setup_folders(LOG_DIR, config, clear=True)
                with open(os.path.join(LOG_DIR, f"config.json"), 'w') as f:
                    json.dump(vars(config), f, indent=4)   
            
            print("\n" + "="*30 + " Configuration " + "="*30)
            print(json.dumps(vars(config), indent=4))
            print("="*75 + "\n")

        if "T1DMS" in config.dataset:
            train_dataset, validate_dataset, test_dataset = prepare_T1DMS_data(
                dataset_str=config.dataset, root_dir=main_path, 
                seq_length=config.seq_len, label_length=config.label_len, pred_length=config.pred_len, 
                use_delta=config.use_delta, sensor_sampling=config.sensor_sampling, gaussian=True)
        elif config.dataset == "OhioT1DM":
            train_dataset, validate_dataset, test_dataset, test_patients, scalar = prepare_Ohio_data(
                root_dir=main_path, seq_length=config.seq_len, label_length=config.label_len, pred_length=config.pred_len, use_delta=config.use_delta)
            zeroshot_dataset, DiaTrend_zeroshot_patients = prepare_DiaTrend_zeroshot_data(
                root_dir=main_path, seq_length=config.seq_len, label_length=config.label_len, pred_length=config.pred_len, 
                external_mean=scalar.mean, external_std=scalar.std, use_delta=config.use_delta)
        elif config.dataset == "DiaTrend":
            train_dataset, validate_dataset, test_dataset, test_patients = prepare_DiaTrend_data(
                root_dir=main_path, seq_length=config.seq_len, label_length=config.label_len, pred_length=config.pred_len, use_delta=config.use_delta)

        train_dataloader = DataLoader(train_dataset, config.batch_size, shuffle=True)
        validate_dataloader = DataLoader(validate_dataset, config.batch_size, shuffle=False)
        train(config, LOG_DIR, train_dataloader, validate_dataloader, device, model)

        if test_dataset is not None:
            test_dataloader = DataLoader(test_dataset, config.batch_size, shuffle=False)
            if config.dataset == "OhioT1DM":
                zeroshot_dataloader = DataLoader(zeroshot_dataset, config.batch_size, shuffle=False)
                eval_per_subject(config, LOG_DIR, model, device, test_dataloader, test_patients, 
                                 zeroshot_dataloader=zeroshot_dataloader, zeroshot_patients=DiaTrend_zeroshot_patients)

            elif config.dataset == "DiaTrend":
                eval_per_subject(config, LOG_DIR, model, device, test_dataloader, test_patients)

    if test_dataset is not None:
        eval_result_dir = os.path.join(LOG_DIR, config.eval_result_path.lstrip('/'))
        summary_specs = [
            ('test', test_patients, 'result_mean_test.csv'),
        ]
        if config.dataset == 'OhioT1DM':
            summary_specs.append(('zeroshot', DiaTrend_zeroshot_patients, 'result_mean_zeroshot.csv'))

        for result_prefix, patient_list, mean_filename in summary_specs:
            eval_mean_csv_path = os.path.join(eval_result_dir, mean_filename)
            subject_files = sorted(glob.glob(os.path.join(eval_result_dir, f'result_subject_{result_prefix}_*.csv')))
            if len(subject_files) == 0:
                print(f"\nNo {result_prefix} subject result files found in {eval_result_dir}.")
                continue

            patient_rows = []
            for subject_file in subject_files:
                patient_id = os.path.splitext(os.path.basename(subject_file))[0].replace(f'result_subject_{result_prefix}_', '')
                df_subject = pd.read_csv(subject_file)

                patient_row = {'patient_id': patient_id}

                for ph in config.PH:
                    rmse_values = df_subject[f'RMSE_{ph}']
                    mae_values = df_subject[f'MAE_{ph}']
                    rmse_mean = rmse_values.mean()
                    rmse_std = rmse_values.std() if len(rmse_values) > 1 else 0.0
                    mae_mean = mae_values.mean()
                    mae_std = mae_values.std() if len(mae_values) > 1 else 0.0
                    patient_row[f'RMSE_{ph}'] = f"{rmse_mean:.4f}±{rmse_std:.4f}"
                    patient_row[f'MAE_{ph}'] = f"{mae_mean:.4f}±{mae_std:.4f}"

                rmse_all_vals = df_subject["RMSE_all"]
                mae_all_vals = df_subject["MAE_all"]
                rmse_all_mean = rmse_all_vals.mean()
                rmse_all_std = rmse_all_vals.std() if len(rmse_all_vals) > 1 else 0.0
                mae_all_mean = mae_all_vals.mean()
                mae_all_std = mae_all_vals.std() if len(mae_all_vals) > 1 else 0.0
                patient_row["RMSE_all"] = f"{rmse_all_mean:.4f}±{rmse_all_std:.4f}"
                patient_row["MAE_all"] = f"{mae_all_mean:.4f}±{mae_all_std:.4f}"

                patient_rows.append(patient_row)

            summary_rows = list(patient_rows)

            overall_row = {'patient_id': 'All_Patients'}

            for ph in config.PH:
                rmse_patient_means = [float(row[f'RMSE_{ph}'].split('±')[0]) for row in patient_rows]
                mae_patient_means = [float(row[f'MAE_{ph}'].split('±')[0]) for row in patient_rows]

                rmse_mean = pd.Series(rmse_patient_means).mean()
                rmse_std = pd.Series(rmse_patient_means).std() if len(rmse_patient_means) > 1 else 0.0
                mae_mean = pd.Series(mae_patient_means).mean()
                mae_std = pd.Series(mae_patient_means).std() if len(mae_patient_means) > 1 else 0.0
                overall_row[f'RMSE_{ph}'] = f"{rmse_mean:.4f}±{rmse_std:.4f}"
                overall_row[f'MAE_{ph}'] = f"{mae_mean:.4f}±{mae_std:.4f}"

            # 【新增2】全体受试者：整段序列 RMSE_all / MAE_all 总平均
            rmse_all_patient_means = [float(row["RMSE_all"].split('±')[0]) for row in patient_rows]
            mae_all_patient_means = [float(row["MAE_all"].split('±')[0]) for row in patient_rows]

            rmse_all_mean_total = pd.Series(rmse_all_patient_means).mean()
            rmse_all_std_total = pd.Series(rmse_all_patient_means).std() if len(rmse_all_patient_means) > 1 else 0.0
            mae_all_mean_total = pd.Series(mae_all_patient_means).mean()
            mae_all_std_total = pd.Series(mae_all_patient_means).std() if len(mae_all_patient_means) > 1 else 0.0

            overall_row["RMSE_all"] = f"{rmse_all_mean_total:.4f}±{rmse_all_std_total:.4f}"
            overall_row["MAE_all"] = f"{mae_all_mean_total:.4f}±{mae_all_std_total:.4f}"

            summary_rows.append(overall_row)

            columns_order = (
                ['patient_id'] 
                + [f'{m}_{ph}' for ph in config.PH for m in ['RMSE', 'MAE']]
                + ['RMSE_all', 'MAE_all']
            )
            mean_df = pd.DataFrame(summary_rows)[columns_order]
            mean_df.to_csv(eval_mean_csv_path, index=False)
            print(f"\nAverage results for {result_prefix} set by patient and overall:")
            print(mean_df.to_string(index=False))
            

def set_model_parameters(config):
    """
    Instantiate the corresponding model and parameters according to config.model_name.
    Returns (config, model, device)
    """
    model = None
    device = torch.device(config.device)
    seq_len = int(config.seq_len/config.sensor_sampling)
    pred_len = int(config.pred_len/config.sensor_sampling)

    if config.model_name == 'Glucoformer0':
        from models.Glucoformer0.Glucoformer import Glucoformer
        from models.Glucoformer0.Glucoformer_parameters import get_Glucoformer_parser
        
        if config.experiment_dir is None:
            override_args = getattr(config, 'override_args', None)
            config = merge_options(config, get_Glucoformer_parser(), args=override_args)
        
        seg_len = int(config.seg_len/config.sensor_sampling)
        model = Glucoformer(
                data_dim=config.data_dim, in_len=seq_len, out_len=pred_len, seg_len=seg_len, output_size=config.c_out,
                factor=config.factor, d_model=config.d_model, d_ff=config.d_ff, n_heads=config.n_heads,
                e_layers=config.e_layers, dropout=config.dropout).to(device)
    
    elif config.model_name == 'Glucoformer_app':
        from models.Glucoformer_app.Glucoformer import Glucoformer
        from models.Glucoformer_app.Glucoformer_parameters import get_Glucoformer_parser
        
        if config.experiment_dir is None:
            override_args = getattr(config, 'override_args', None)
            config = merge_options(config, get_Glucoformer_parser(), args=override_args)
        
        seg_len = int(config.seg_len/config.sensor_sampling)
        model = Glucoformer(
                data_dim=config.data_dim, in_len=seq_len, out_len=pred_len, seg_len=seg_len, output_size=config.c_out,
                factor=config.factor, d_model=config.d_model, d_ff=config.d_ff, n_heads=config.n_heads,
            e_layers=config.e_layers, dropout=config.dropout,).to(device)

    elif config.model_name == 'Glucoformer':
        from models.Glucoformer.Glucoformer import Glucoformer
        from models.Glucoformer.Glucoformer_parameters import get_Glucoformer_parser
        
        if config.experiment_dir is None:
            override_args = getattr(config, 'override_args', None)
            config = merge_options(config, get_Glucoformer_parser(), args=override_args)
        
        seg_len = int(config.seg_len/config.sensor_sampling)
        model = Glucoformer(
                data_dim=config.data_dim, in_len=seq_len, out_len=pred_len, seg_len=seg_len, output_size=config.c_out,
                factor=config.factor, d_model=config.d_model, d_ff=config.d_ff, n_heads=config.n_heads,
            e_layers=config.e_layers, dropout=config.dropout, 
            use_sse=config.use_sse, use_etsa=config.use_etsa, use_decoder_self_attn=config.use_decoder_self_attn).to(device)

    elif config.model_name == 'Crossformer':
        from models.Crossformer.cross_former import Crossformer
        from models.Crossformer.Crossformer_parameters import get_Crossformer_parser
        if config.experiment_dir is None:
            config = merge_options(config, get_Crossformer_parser())
        seg_len = int(config.seg_len/config.sensor_sampling)
        model = Crossformer(data_dim=config.data_dim, in_len=seq_len, out_len=pred_len, seg_len=seg_len,  output_size=config.c_out,
                            factor=config.factor, d_model=config.d_model, d_ff=config.d_ff, 
                            n_heads=config.n_heads, e_layers=config.e_layers, dropout=config.dropout).to(device)

    elif config.model_name == 'PatchTST':
        from models.PatchTST.PatchTST import PatchTST
        from models.PatchTST.PatchTST_parameters import get_PatchTST_parser
        if config.experiment_dir is None:
            config = merge_options(config, get_PatchTST_parser())
        model = PatchTST(config).to(device)

    elif config.model_name == 'TimeXer':
        from models.TimeXer.TimeXer import TimeXer
        from models.TimeXer.TimeXer_parameters import get_TimeXer_parser
        if config.experiment_dir is None:
            config = merge_options(config, get_TimeXer_parser())
        model = TimeXer(config).to(device)

    elif config.model_name == 'DLinear':
        from models.DLinear.DLinear_model import DLinear
        from models.DLinear.DLinear_parameters import get_DLinear_parser
        if config.experiment_dir is None:
            config = merge_options(config, get_DLinear_parser())
        model = DLinear(config).to(device)

    elif config.model_name == 'Informer':
        from models.Informer.model import Informer, InformerStack
        from models.Informer.Informer_parameters import get_Informer_parser
        if config.experiment_dir is None:
            config = merge_options(config, get_Informer_parser())
        if config.model_name == 'Informer':
            model = Informer(
                enc_in=config.enc_in, dec_in=config.dec_in, c_out=config.c_out, out_len=pred_len,
                factor=config.factor, d_model=config.d_model, n_heads=config.n_heads,
                e_layers=config.e_layers, d_layers=config.d_layers, d_ff=config.d_ff, dropout=config.dropout
            ).to(device)
        else:
            model = InformerStack(
                enc_in=config.enc_in, dec_in=config.dec_in, c_out=config.c_out, out_len=pred_len,
                factor=config.factor, d_model=config.d_model, n_heads=config.n_heads,
                d_layers=config.d_layers, d_ff=config.d_ff, dropout=config.dropout
            ).to(device)

    elif config.model_name == 'Transformer':
        from models.Transformer.Transformer_model import Transformer
        from models.Transformer.Transformer_parameters import get_Transformer_parser
        if config.experiment_dir is None:
            config = merge_options(config, get_Transformer_parser())
        model = Transformer(input_size=config.enc_in, pred_length=pred_len, d_model=config.d_model, device=config.device, nhead=config.n_heads, 
                            num_encoder_layers=config.e_layers, num_decoder_layers=config.d_layers, dim_feedforward=config.d_ff, output_size=1, 
                            dropout=config.dropout).to(device)

    elif config.model_name == 'LSTM':
        from models.LSTM.LSTM_model import LSTM
        from models.LSTM.LSTM_parameters import get_LSTM_parser
        if config.experiment_dir is None:
            config = merge_options(config, get_LSTM_parser())
        model = LSTM(input_size=config.data_dim, hidden_size=config.d_model, num_layers=config.e_layers, 
                    forecast_window=pred_len, output_size=config.c_out, dropout=config.dropout).to(device)

    elif config.model_name == 'GRU':
        from models.GRU.GRU_model import GRU
        from models.GRU.GRU_parameters import get_GRU_parser
        if config.experiment_dir is None:
            config = merge_options(config, get_GRU_parser())
        model = GRU(input_size=config.data_dim, hidden_size=config.d_model, num_layers=config.e_layers, 
                    forecast_window=pred_len, output_size=config.c_out).to(device)

    else:
        raise ValueError(f"Unknown model name: {config.model_name}")

    return config, model, device


def train(config, LOG_DIR, train_dataloader, validate_dataloader, device, model):
    
    model_path = os.path.join(LOG_DIR+config.model_save_path, f'{config.model_name}_seed{config.seed}.pth')
    loss_record_csv_path = os.path.join(LOG_DIR+config.loss_record_path, f'loss_record_{config.seed}.csv')
    fig_save_path = os.path.join(LOG_DIR+config.loss_record_path, f'plot_loss_record_{config.seed}.png')
    validate_min_loss = float('inf')
    early_stop_cnt = 0

    if config.pretrain_model_pth is not None and config.pretrain_model_pth != 'None':
        model.load_state_dict(torch.load(config.pretrain_model_pth))
        print("Loaded pre-trained model weights successfully!\n")
    
    criterion1 = nn.MSELoss()
    criterion2 = nn.L1Loss()
    if "T1DMS" in config.dataset:
        criterion3 = MultiHorizonMSELoss(pred_len=int(config.pred_len/config.sensor_sampling), mode='power', beta=2)
    else:
        criterion3 = MultiHorizonMSELoss(pred_len=int(config.pred_len/config.sensor_sampling), mode='power', beta=0.7)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.step_size, config.pre_gamma)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=config.pre_gamma)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train_epochs * len(train_dataloader))

    for epoch in range(1, config.train_epochs+1):  
        total_train_MSE_loss = 0
        total_train_MAE_loss = 0
        total_validate_MSE_loss = 0
        total_validate_MAE_loss = 0

        print(f"{config.model_name} model is training... Epoch: {epoch}/{config.train_epochs}")
        current_time = datetime.datetime.now()
        model.train()
        pbar = tqdm(train_dataloader)
        for _, (encoder_input, encoder_input_mark, decoder_input ,decoder_input_mark, tgt, _) in enumerate (pbar):
            encoder_input = encoder_input.to(device)
            encoder_input_mark = encoder_input_mark.to(device)
            decoder_input = decoder_input.to(device)
            decoder_input_mark = decoder_input_mark.to(device)
            tgt = tgt.to(device)
            last_lr = optimizer.param_groups[0]['lr'] 

            optimizer.zero_grad()
            output = model(encoder_input, encoder_input_mark, decoder_input, decoder_input_mark)

            train_dataset = train_dataloader.dataset
            if hasattr(train_dataset, "datasets"):
                inverse_transform_func = train_dataset.datasets[0].inverse_transform
                mean = train_dataset.datasets[0].mean[2]
                std = train_dataset.datasets[0].std[2]
            else:
                inverse_transform_func = train_dataset.inverse_transform
                mean = train_dataset.mean[2]
                std = train_dataset.std[2]

            if config.use_delta:
                # Compute delta-based and absolute BG prediction losses and combine them for training
                bg_current_norm = encoder_input[:, -1, -1].unsqueeze(1)
                mean = torch.tensor(mean, device=output.device)
                std = torch.tensor(std, device=output.device)
                bg_current_real = bg_current_norm * std + mean
                delta_pred_real = output * std
                delta_true_real = tgt * std
                bg_pred_real = bg_current_real + delta_pred_real.squeeze(-1)
                bg_true_real = bg_current_real + delta_true_real.squeeze(-1)
                train_MSE_loss_inverse = criterion1(bg_pred_real, bg_true_real)
                train_MAE_loss_inverse = criterion2(bg_pred_real, bg_true_real)
                if getattr(config, 'weight_RMSE', True):
                    loss_delta = criterion3(delta_pred_real, delta_true_real)
                else:
                    loss_delta = criterion1(delta_pred_real, delta_true_real)
                train_loss = loss_delta #+ 0.5 * loss_BG
                train_loss.backward()
                optimizer.step()
                scheduler.step()
            else:
                # Standard loss on normalized data
                train_loss = criterion1(output, tgt)
                output_last = output[:, :, :]
                tgt_last = tgt[:, :, :]
                output_numpy = output_last.reshape(-1, 1).detach().cpu().numpy()
                tgt_numpy = tgt_last.reshape(-1, 1).detach().cpu().numpy()
                train_output_inverse = torch.tensor(inverse_transform_func(output_numpy), dtype=torch.float32)
                train_target_inverse = torch.tensor(inverse_transform_func(tgt_numpy), dtype=torch.float32)
                train_MSE_loss_inverse = criterion1(train_output_inverse, train_target_inverse)
                train_MAE_loss_inverse = criterion2(train_output_inverse, train_target_inverse)
                train_loss.backward()
                optimizer.step()
                scheduler.step()

            total_train_MSE_loss += train_MSE_loss_inverse.item()
            total_train_MAE_loss += train_MAE_loss_inverse.item()

        model.eval()
        with torch.no_grad():
            pbar = tqdm(validate_dataloader)
            for _, (encoder_input, encoder_input_mark, decoder_input ,decoder_input_mark, tgt, _) in enumerate (pbar):
                encoder_input = encoder_input.to(device)
                encoder_input_mark = encoder_input_mark.to(device)
                decoder_input = decoder_input.to(device)
                decoder_input_mark = decoder_input_mark.to(device)
                tgt = tgt.to(device)

                output = model(encoder_input, encoder_input_mark, decoder_input, decoder_input_mark)

                validate_dataset = validate_dataloader.dataset
                if hasattr(validate_dataset, "datasets"):
                    inverse_transform_func = validate_dataset.datasets[0].inverse_transform
                    mean = validate_dataset.datasets[0].mean[2]
                    std = validate_dataset.datasets[0].std[2]
                else:
                    inverse_transform_func = validate_dataset.inverse_transform
                    mean = validate_dataset.mean[2]
                    std = validate_dataset.std[2]
                
                if config.use_delta:
                    # Compute delta-based and absolute BG prediction losses and combine them for training
                    bg_current_norm = encoder_input[:, -1, -1].unsqueeze(1)
                    mean = torch.tensor(mean, device=output.device)
                    std = torch.tensor(std, device=output.device)
                    bg_current_real = bg_current_norm * std + mean
                    delta_pred_real = output * std
                    delta_true_real = tgt * std
                    bg_pred_real = bg_current_real + delta_pred_real.squeeze(-1)
                    bg_true_real = bg_current_real + delta_true_real.squeeze(-1)
                    validate_MSE_loss_inverse = criterion1(bg_pred_real, bg_true_real)
                    validate_MAE_loss_inverse = criterion2(bg_pred_real, bg_true_real)
                else:
                    output_last = output[:, :, :]
                    tgt_last = tgt[:, :, :]
                    output_numpy = output_last.reshape(-1, 1).detach().cpu().numpy()
                    tgt_numpy = tgt_last.reshape(-1, 1).detach().cpu().numpy()
                    validate_output_inverse = torch.tensor(inverse_transform_func(output_numpy), dtype=torch.float32)
                    validate_target_inverse = torch.tensor(inverse_transform_func(tgt_numpy), dtype=torch.float32)
                    validate_MSE_loss_inverse = criterion1(validate_output_inverse, validate_target_inverse)
                    validate_MAE_loss_inverse = criterion2(validate_output_inverse, validate_target_inverse)

                total_validate_MSE_loss += validate_MSE_loss_inverse.item()
                total_validate_MAE_loss += validate_MAE_loss_inverse.item()

        avg_train_RMSE_loss = math.sqrt(total_train_MSE_loss/len(train_dataloader))
        avg_train_MAE_loss = total_train_MAE_loss/len(train_dataloader)
        avg_validate_RMSE_loss = math.sqrt(total_validate_MSE_loss /len(validate_dataloader))
        avg_validate_MAE_loss = total_validate_MAE_loss /len(validate_dataloader)

        if avg_validate_RMSE_loss < validate_min_loss:
            # Save model if your model improved
            validate_min_loss = avg_validate_RMSE_loss
            torch.save(model.state_dict(), model_path)
            early_stop_cnt = 0
            print('Saving best model (epoch = {:4d}, loss = {:.4f})'.format(epoch, validate_min_loss))
        else:
            early_stop_cnt += 1

        current_data = {
            'timestamp': current_time,
            'epoch': epoch,
            'train_RMSE': avg_train_RMSE_loss,
            'train_MAE': avg_train_MAE_loss,
            'val_RMSE': avg_validate_RMSE_loss,
            'val_MAE': avg_validate_MAE_loss,
            'learning_rate': last_lr,
            'early_stop_cnt': early_stop_cnt
        }
        df_new = pd.DataFrame([current_data])
        df_new.to_csv(loss_record_csv_path, mode='a', header=not os.path.exists(loss_record_csv_path), index=False)
        record = (
            f"Learning Rate: {last_lr}\n"
            f"Train_RMSE_loss: {avg_train_RMSE_loss:.6f}, Train_MAE_loss: {avg_train_MAE_loss:.6f}\n"
            f"Val_RMSE_loss: {avg_validate_RMSE_loss:.6f}, Val_MAE_loss: {avg_validate_MAE_loss:.6f}\n")
        print(record)

        if epoch % 1 == 0:
            plot_loss_curve(loss_record_csv_path, fig_save_path)
        
        if early_stop_cnt > config.patience:
            break
    
    print('Finished training after {} epochs, validate_min_loss: {:.6f})'.format(epoch, validate_min_loss))


# def eval_per_subject(config, LOG_DIR, model, device, test_dataloader, test_patients, 
#                      zeroshot_dataloader=None, zeroshot_patients=None, PH=[30, 60, 90, 120], model_path=None):
#     if model_path is None:
#         model_path = os.path.join(LOG_DIR+config.model_save_path, f'{config.model_name}_seed{config.seed}.pth')

#     criterion1 = nn.MSELoss()
#     criterion2 = nn.L1Loss()
#     model.load_state_dict(torch.load(model_path))

#     print(f"Evaluating {config.model_name} model by subject, please wait...")
#     model.eval()
    
#     test_dataset = test_dataloader.dataset
#     # 拆分两套独立受试者数据集：普通测试集、零样本外部集
#     test_patient_datasets = []
#     zs_patient_datasets = []

#     # ===================== 按数据集自动分配数据 =====================
#     if config.dataset == "OhioT1DM" and hasattr(test_dataset, "datasets"):
#         # 构建普通测试集受试者数据
#         for i, patient in enumerate(test_patients):
#             test_patient_datasets.append((patient, torch.utils.data.ConcatDataset([
#                 test_dataset.datasets[2 * i], 
#                 test_dataset.datasets[2 * i + 1]
#             ])))
#         # 有零样本数据则构建外部测试集
#         if zeroshot_dataloader is not None and zeroshot_patients is not None:
#             zeroshot_dataset = zeroshot_dataloader.dataset  
#             for x, zeroshot_patient in enumerate(zeroshot_patients):
#                 zs_patient_datasets.append((zeroshot_patient, zeroshot_dataset.datasets[x]))
                
#     elif config.dataset == "DiaTrend" and hasattr(test_dataset, "datasets"):
#         # 其他数据集仅构建普通测试集
#         for i, patient in enumerate(test_patients):
#             test_patient_datasets.append((patient, test_dataset.datasets[i]))

#     # ===================== 通用评估函数（内部复用，无结构改动） =====================
#     def run_eval(ds_list, prefix):
#         results = []
#         for patient_id, dataset in ds_list:
#             ph_metrics = dict(**{f'RMSE_{ph}': 0 for ph in PH}, **{f'MAE_{ph}': 0 for ph in PH})
            
#             sub_dataloader = torch.utils.data.DataLoader(
#                 dataset,
#                 batch_size=test_dataloader.batch_size,
#                 shuffle=False, 
#                 num_workers=test_dataloader.num_workers,
#                 drop_last=False
#             )
            
#             with torch.no_grad():
#                 for _, (encoder_input, encoder_input_mark, decoder_input, decoder_input_mark, tgt, _) in enumerate(sub_dataloader):
#                     encoder_input = encoder_input.to(device)
#                     encoder_input_mark = encoder_input_mark.to(device)
#                     decoder_input = decoder_input.to(device)
#                     decoder_input_mark = decoder_input_mark.to(device)
#                     tgt = tgt.to(device)
                    
#                     output = model(encoder_input, encoder_input_mark, decoder_input, decoder_input_mark)
                    
#                     for ph in PH:
#                         idx = int(ph / config.sensor_sampling) - 1
#                         output_ph = output[:, idx, :]
#                         tgt_ph = tgt[:, idx, :]

#                         test_dataset = test_dataloader.dataset
#                         if hasattr(test_dataset, "datasets"):
#                             inverse_transform_func = test_dataset.datasets[0].inverse_transform
#                             mean = test_dataset.datasets[0].mean[2]
#                             std = test_dataset.datasets[0].std[2]
#                         else:
#                             inverse_transform_func = test_dataset.inverse_transform
#                             mean = test_dataset.mean[2]
#                             std = test_dataset.std[2]

#                         if config.use_delta:
#                             bg_current_norm = encoder_input[:, -1, -1].unsqueeze(1)
#                             mean_t = torch.tensor(mean, device=output.device)
#                             std_t = torch.tensor(std, device=output.device)
#                             bg_current_real = bg_current_norm * std_t + mean_t
#                             delta_pred_real = output_ph * std_t
#                             delta_true_real = tgt_ph * std_t
#                             bg_pred_real = bg_current_real + delta_pred_real.squeeze(-1)
#                             bg_true_real = bg_current_real + delta_true_real.squeeze(-1)
#                             mse = criterion1(bg_pred_real, bg_true_real).item()
#                             mae = criterion2(bg_pred_real, bg_true_real).item()
#                         else:
#                             output_numpy = output_ph.reshape(-1, 1).detach().cpu().numpy()
#                             tgt_numpy = tgt_ph.reshape(-1, 1).detach().cpu().numpy()
#                             output_inv = torch.tensor(inverse_transform_func(output_numpy), dtype=torch.float32)
#                             tgt_inv = torch.tensor(inverse_transform_func(tgt_numpy), dtype=torch.float32)
#                             mse = criterion1(output_inv, tgt_inv).item()
#                             mae = criterion2(output_inv, tgt_inv).item()

#                         ph_metrics[f'RMSE_{ph}'] += mse
#                         ph_metrics[f'MAE_{ph}'] += mae

#             ph_metrics['run_seed'] = config.seed
#             ph_metrics['patient_id'] = patient_id
#             for ph in PH:
#                 ph_metrics[f'RMSE_{ph}'] = math.sqrt(ph_metrics[f'RMSE_{ph}'] / len(sub_dataloader))
#                 ph_metrics[f'MAE_{ph}'] /= len(sub_dataloader)
                
#             # 文件命名使用对应前缀，区分两类结果
#             subject_csv_path = os.path.join(LOG_DIR+config.eval_result_path, f'result_subject_{prefix}_{patient_id}.csv')
#             columns_order = ['run_seed'] + [f'{m}_{ph}' for ph in PH for m in ['RMSE', 'MAE']]
#             df_subject = pd.DataFrame([ph_metrics])[columns_order]
#             df_subject.to_csv(subject_csv_path, mode='a', header=not os.path.exists(subject_csv_path), index=False)
            
#             results.append(ph_metrics)
#         return results

#     # ===================== 自动执行评估 =====================
#     # 1. 始终运行普通测试集
#     test_results = run_eval(test_patient_datasets, "test")
#     # 打印测试集汇总
#     if test_results:
#         columns_order_print = ['run_seed', 'patient_id'] + [f'{m}_{ph}' for ph in PH for m in ['RMSE', 'MAE']]
#         df_test = pd.DataFrame(test_results)[columns_order_print]
#         print(f"Evaluation completed for seed {config.seed}. Average across test set subjects:")
#         mean_rows = []
#         for ph in PH:
#             rmse_mean = df_test[f'RMSE_{ph}'].mean()
#             rmse_std = df_test[f'RMSE_{ph}'].std() if len(df_test) > 1 else 0.0
#             mae_mean = df_test[f'MAE_{ph}'].mean()
#             mae_std = df_test[f'MAE_{ph}'].std() if len(df_test) > 1 else 0.0
#             mean_rows.append({
#                 'PH': ph,
#                 'RMSE': f"{rmse_mean:.4f}±{rmse_std:.4f}",
#                 'MAE': f"{mae_mean:.4f}±{mae_std:.4f}"
#             })
#         print(pd.DataFrame(mean_rows, columns=["PH", "RMSE", "MAE"]).to_string(index=False))

#     # 2. 仅 OhioT1DM 且存在零样本数据时，运行外部零样本集
#     if config.dataset == "OhioT1DM" and zs_patient_datasets:
#         zs_results = run_eval(zs_patient_datasets, "zeroshot")
#         if zs_results:
#             columns_order_print = ['run_seed', 'patient_id'] + [f'{m}_{ph}' for ph in PH for m in ['RMSE', 'MAE']]
#             df_zs = pd.DataFrame(zs_results)[columns_order_print]
#             print(f"\nEvaluation completed for seed {config.seed}. Average across zero-shot set subjects:")
#             mean_rows_zs = []
#             for ph in PH:
#                 rmse_mean = df_zs[f'RMSE_{ph}'].mean()
#                 rmse_std = df_zs[f'RMSE_{ph}'].std() if len(df_zs) > 1 else 0.0
#                 mae_mean = df_zs[f'MAE_{ph}'].mean()
#                 mae_std = df_zs[f'MAE_{ph}'].std() if len(df_zs) > 1 else 0.0
#                 mean_rows_zs.append({
#                     'PH': ph,
#                     'RMSE': f"{rmse_mean:.4f}±{rmse_std:.4f}",
#                     'MAE': f"{mae_mean:.4f}±{mae_std:.4f}"
#                 })
#             print(pd.DataFrame(mean_rows_zs, columns=["PH", "RMSE", "MAE"]).to_string(index=False))


def eval_per_subject(config, LOG_DIR, model, device, test_dataloader, test_patients, 
                     zeroshot_dataloader=None, zeroshot_patients=None, PH=[30, 60, 90, 120], model_path=None):
    if model_path is None:
        model_path = os.path.join(LOG_DIR+config.model_save_path, f'{config.model_name}_seed{config.seed}.pth')

    criterion1 = nn.MSELoss()
    criterion2 = nn.L1Loss()
    model.load_state_dict(torch.load(model_path))

    print(f"Evaluating {config.model_name} model by subject, please wait...")
    model.eval()
    
    test_dataset = test_dataloader.dataset
    test_patient_datasets = []
    zs_patient_datasets = []

    if config.dataset == "OhioT1DM" and hasattr(test_dataset, "datasets"):
        for i, patient in enumerate(test_patients):
            test_patient_datasets.append((patient, torch.utils.data.ConcatDataset([
                test_dataset.datasets[2 * i], 
                test_dataset.datasets[2 * i + 1]
            ])))
        if zeroshot_dataloader is not None and zeroshot_patients is not None:
            zeroshot_dataset = zeroshot_dataloader.dataset  
            for x, zeroshot_patient in enumerate(zeroshot_patients):
                zs_patient_datasets.append((zeroshot_patient, zeroshot_dataset.datasets[x]))
                
    elif config.dataset == "DiaTrend" and hasattr(test_dataset, "datasets"):
        for i, patient in enumerate(test_patients):
            test_patient_datasets.append((patient, test_dataset.datasets[i]))


    def run_eval(ds_list, prefix):
        results = []
        for patient_id, dataset in ds_list:
            ph_metrics = dict(**{f'RMSE_{ph}': 0 for ph in PH}, **{f'MAE_{ph}': 0 for ph in PH})
            ph_metrics["RMSE_all"] = 0.0   
            ph_metrics["MAE_all"] = 0.0    
            
            sub_dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=test_dataloader.batch_size,
                shuffle=False, 
                num_workers=test_dataloader.num_workers,
                drop_last=False
            )
            
            with torch.no_grad():
                for _, (encoder_input, encoder_input_mark, decoder_input, decoder_input_mark, tgt, _) in enumerate(sub_dataloader):
                    encoder_input = encoder_input.to(device)
                    encoder_input_mark = encoder_input_mark.to(device)
                    decoder_input = decoder_input.to(device)
                    decoder_input_mark = decoder_input_mark.to(device)
                    tgt = tgt.to(device)
                    
                    output = model(encoder_input, encoder_input_mark, decoder_input, decoder_input_mark)
                    
                    for ph in PH:
                        idx = int(ph / config.sensor_sampling) - 1
                        output_ph = output[:, idx, :]
                        tgt_ph = tgt[:, idx, :]

                        test_dataset = test_dataloader.dataset
                        if hasattr(test_dataset, "datasets"):
                            inverse_transform_func = test_dataset.datasets[0].inverse_transform
                            mean = test_dataset.datasets[0].mean[2]
                            std = test_dataset.datasets[0].std[2]
                        else:
                            inverse_transform_func = test_dataset.inverse_transform
                            mean = test_dataset.mean[2]
                            std = test_dataset.std[2]

                        if config.use_delta:
                            bg_current_norm = encoder_input[:, -1, -1].unsqueeze(1)
                            mean_t = torch.tensor(mean, device=output.device)
                            std_t = torch.tensor(std, device=output.device)
                            bg_current_real = bg_current_norm * std_t + mean_t
                            delta_pred_real = output_ph * std_t
                            delta_true_real = tgt_ph * std_t
                            bg_pred_real = bg_current_real + delta_pred_real.squeeze(-1)
                            bg_true_real = bg_current_real + delta_true_real.squeeze(-1)
                            mse = criterion1(bg_pred_real, bg_true_real).item()
                            mae = criterion2(bg_pred_real, bg_true_real).item()
                        else:
                            output_numpy = output_ph.reshape(-1, 1).detach().cpu().numpy()
                            tgt_numpy = tgt_ph.reshape(-1, 1).detach().cpu().numpy()
                            output_inv = torch.tensor(inverse_transform_func(output_numpy), dtype=torch.float32)
                            tgt_inv = torch.tensor(inverse_transform_func(tgt_numpy), dtype=torch.float32)
                            mse = criterion1(output_inv, tgt_inv).item()
                            mae = criterion2(output_inv, tgt_inv).item()

                        ph_metrics[f'RMSE_{ph}'] += mse
                        ph_metrics[f'MAE_{ph}'] += mae

                    test_dataset = test_dataloader.dataset
                    if hasattr(test_dataset, "datasets"):
                        inverse_transform_func = test_dataset.datasets[0].inverse_transform
                        mean = test_dataset.datasets[0].mean[2]
                        std = test_dataset.datasets[0].std[2]
                    else:
                        inverse_transform_func = test_dataset.inverse_transform
                        mean = test_dataset.mean[2]
                        std = test_dataset.std[2]

                    if config.use_delta:
                        bg_current_norm = encoder_input[:, -1, -1].unsqueeze(1)
                        mean_t = torch.tensor(mean, device=output.device)
                        std_t = torch.tensor(std, device=output.device)
                        bg_current_real = bg_current_norm * std_t + mean_t
                        delta_pred_real = output * std_t
                        delta_true_real = tgt * std_t
                        bg_pred_real_all = bg_current_real + delta_pred_real.squeeze(-1)
                        bg_true_real_all = bg_current_real + delta_true_real.squeeze(-1)
                        mse_all = criterion1(bg_pred_real_all, bg_true_real_all).item()
                        mae_all = criterion2(bg_pred_real_all, bg_true_real_all).item()
                    else:
                        output_flat = output.reshape(-1, 1).detach().cpu().numpy()
                        tgt_flat = tgt.reshape(-1, 1).detach().cpu().numpy()
                        output_inv_all = torch.tensor(inverse_transform_func(output_flat), dtype=torch.float32)
                        tgt_inv_all = torch.tensor(inverse_transform_func(tgt_flat), dtype=torch.float32)
                        mse_all = criterion1(output_inv_all, tgt_inv_all).item()
                        mae_all = criterion2(tgt_inv_all, tgt_inv_all).item()

                    ph_metrics["RMSE_all"] += mse_all
                    ph_metrics["MAE_all"] += mae_all

            ph_metrics['run_seed'] = config.seed
            ph_metrics['patient_id'] = patient_id

            for ph in PH:
                ph_metrics[f'RMSE_{ph}'] = math.sqrt(ph_metrics[f'RMSE_{ph}'] / len(sub_dataloader))
                ph_metrics[f'MAE_{ph}'] /= len(sub_dataloader)
            
            ph_metrics["RMSE_all"] = math.sqrt(ph_metrics["RMSE_all"] / len(sub_dataloader))
            ph_metrics["MAE_all"] /= len(sub_dataloader)
                
            subject_csv_path = os.path.join(LOG_DIR+config.eval_result_path, f'result_subject_{prefix}_{patient_id}.csv')
            columns_order = (
                ['run_seed'] 
                + [f'{m}_{ph}' for ph in PH for m in ['RMSE', 'MAE']]
                + ['RMSE_all', 'MAE_all'] 
            )
            df_subject = pd.DataFrame([ph_metrics])[columns_order]
            df_subject.to_csv(subject_csv_path, mode='a', header=not os.path.exists(subject_csv_path), index=False)
            
            results.append(ph_metrics)
        return results


    test_results = run_eval(test_patient_datasets, "test")
    if test_results:
        columns_order_print = (
            ['run_seed', 'patient_id'] 
            + [f'{m}_{ph}' for ph in PH for m in ['RMSE', 'MAE']]
            + ['RMSE_all', 'MAE_all']
        )
        df_test = pd.DataFrame(test_results)[columns_order_print]
        print(f"Evaluation completed for seed {config.seed}. Average across test set subjects:")
        mean_rows = []
        for ph in PH:
            rmse_mean = df_test[f'RMSE_{ph}'].mean()
            rmse_std = df_test[f'RMSE_{ph}'].std() if len(df_test) > 1 else 0.0
            mae_mean = df_test[f'MAE_{ph}'].mean()
            mae_std = df_test[f'MAE_{ph}'].std() if len(df_test) > 1 else 0.0
            mean_rows.append({
                'PH': ph,
                'RMSE': f"{rmse_mean:.4f}±{rmse_std:.4f}",
                'MAE': f"{mae_mean:.4f}±{mae_std:.4f}"
            })
        rmse_all_mean = df_test["RMSE_all"].mean()
        rmse_all_std = df_test["RMSE_all"].std() if len(df_test) > 1 else 0.0
        mae_all_mean = df_test["MAE_all"].mean()
        mae_all_std = df_test["MAE_all"].std() if len(df_test) > 1 else 0.0
        mean_rows.append({
            'PH': 'All_Steps',
            'RMSE': f"{rmse_all_mean:.4f}±{rmse_all_std:.4f}",
            'MAE': f"{mae_all_mean:.4f}±{mae_all_std:.4f}"
        })
        print(pd.DataFrame(mean_rows, columns=["PH", "RMSE", "MAE"]).to_string(index=False))

    if config.dataset == "OhioT1DM" and zs_patient_datasets:
        zs_results = run_eval(zs_patient_datasets, "zeroshot")
        if zs_results:
            columns_order_print = (
                ['run_seed', 'patient_id'] 
                + [f'{m}_{ph}' for ph in PH for m in ['RMSE', 'MAE']]
                + ['RMSE_all', 'MAE_all']
            )
            df_zs = pd.DataFrame(zs_results)[columns_order_print]
            print(f"\nEvaluation completed for seed {config.seed}. Average across zero-shot set subjects:")
            mean_rows_zs = []

            for ph in PH:
                rmse_mean = df_zs[f'RMSE_{ph}'].mean()
                rmse_std = df_zs[f'RMSE_{ph}'].std() if len(df_zs) > 1 else 0.0
                mae_mean = df_zs[f'MAE_{ph}'].mean()
                mae_std = df_zs[f'MAE_{ph}'].std() if len(df_zs) > 1 else 0.0
                mean_rows_zs.append({
                    'PH': ph,
                    'RMSE': f"{rmse_mean:.4f}±{rmse_std:.4f}",
                    'MAE': f"{mae_mean:.4f}±{mae_std:.4f}"
                })

            rmse_all_mean = df_zs["RMSE_all"].mean()
            rmse_all_std = df_zs["RMSE_all"].std() if len(df_zs) > 1 else 0.0
            mae_all_mean = df_zs["MAE_all"].mean()
            mae_all_std = df_zs["MAE_all"].std() if len(df_zs) > 1 else 0.0
            mean_rows_zs.append({
                'PH': 'All_Steps',
                'RMSE': f"{rmse_all_mean:.4f}±{rmse_all_std:.4f}",
                'MAE': f"{mae_all_mean:.4f}±{mae_all_std:.4f}"
            })
            print(pd.DataFrame(mean_rows_zs, columns=["PH", "RMSE", "MAE"]).to_string(index=False))