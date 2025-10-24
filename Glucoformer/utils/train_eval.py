import torch
from joblib import load
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from utils.tools import *
# from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import math
import random
from Glucose_Data.T1DMS_GlucoseDataset import T1DMS_GlucoseDataset
from Glucose_Data.OhioDataset import *


def pretrain_model(config, model, train_dataloader, validate_dataloader, path_to_save_model, device):
    
    criterion1 = nn.MSELoss()
    criterion2 = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=config.pre_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.step_size, config.pre_gamma)

    # writer = SummaryWriter(f"logs_train_pred{config.pred_len}min；{datetime.now().strftime("%Y-%m-%d；%Hh%Mm%Ss")}/")
    sensor_scaler = load(config.path_to_save_scaler+'sensor_scaler.joblib')
    early_stopping = EarlyStopping(patience=config.patience, pretrain=True, verbose=False, delta=0)

    for epoch in range(1, config.train_epochs+1):  
        total_train_MSE_loss = 0
        total_train_MAE_loss = 0
        total_validate_MSE_loss = 0
        total_validate_MAE_loss = 0
        total_validate_accuracy = 0

        model.train()

        pbar = tqdm(train_dataloader)
        for step, (encoder_input, encoder_input_mark, decoder_input ,decoder_input_mark, tgt, _) in enumerate (pbar):

            encoder_input, encoder_input_mark, decoder_input, decoder_input_mark, tgt = encoder_input.to(device), encoder_input_mark.to(device), decoder_input.to(device), decoder_input_mark.to(device), tgt.to(device)

            optimizer.zero_grad()

            output = model(encoder_input, encoder_input_mark, decoder_input, decoder_input_mark)
            # print(encoder_input.shape, decoder_input.shape, tgt.shape, output.shape)

            train_loss = criterion1(output, tgt)

            train_loss.backward()

            optimizer.step()

            train_output_inverse = torch.tensor(sensor_scaler.inverse_transform(output.reshape(-1,1).detach().cpu().numpy()).reshape(output.shape), dtype=torch.float32)
            train_target_inverse = torch.tensor(sensor_scaler.inverse_transform(tgt.reshape(-1,1).detach().cpu().numpy()).reshape(tgt.shape), dtype=torch.float32)
            train_MSE_loss_inverse = criterion1(train_output_inverse, train_target_inverse)
            train_MAE_loss_inverse = criterion2(train_output_inverse, train_target_inverse)
            total_train_MSE_loss += train_MSE_loss_inverse.item()
            total_train_MAE_loss += train_MAE_loss_inverse.item()

            train_s = "Train ==> [Epoch: {}/{}] - step:{} - train_MSE_loss:{:.8f} - train_MAE_loss:{:.8f}".format(epoch, config.train_epochs, step+1, train_MSE_loss_inverse, train_MAE_loss_inverse)
            pbar.set_description(train_s)


        model.eval()
        with torch.no_grad():

            pbar = tqdm(validate_dataloader)
            for step, (encoder_input, encoder_input_mark, decoder_input ,decoder_input_mark, tgt, _) in enumerate (pbar):

                encoder_input, encoder_input_mark, decoder_input, decoder_input_mark, tgt = encoder_input.to(device), encoder_input_mark.to(device), decoder_input.to(device), decoder_input_mark.to(device), tgt.to(device)

                output = model(encoder_input, encoder_input_mark, decoder_input, decoder_input_mark)

                validate_output_inverse = torch.tensor(sensor_scaler.inverse_transform(output.reshape(-1,1).detach().cpu().numpy()).reshape(output.shape), dtype=torch.float32)
                validate_target_inverse = torch.tensor(sensor_scaler.inverse_transform(tgt.reshape(-1,1).detach().cpu().numpy()).reshape(tgt.shape), dtype=torch.float32)
                validate_MSE_loss_inverse = criterion1(validate_output_inverse, validate_target_inverse)
                validate_MAE_loss_inverse = criterion2(validate_output_inverse, validate_target_inverse)
                total_validate_MSE_loss += validate_MSE_loss_inverse.item()
                total_validate_MAE_loss += validate_MAE_loss_inverse.item()

                correct = (torch.abs(validate_output_inverse - validate_target_inverse) <= config.tolerance).sum().item()
                total_elements = output.numel()  # Total number of elements (16 * 3)
                validate_accuracy = correct / total_elements
                total_validate_accuracy += validate_accuracy

                validate_s = "Validate ==> [Epoch: {}/{}] - step:{} - Val_MSE_loss:{:.8f} - Val_MAE_loss:{:.8f} - Val_accuracy:{:.8f}".format(epoch, config.train_epochs, step+1, validate_MSE_loss_inverse, validate_MAE_loss_inverse, validate_accuracy)
                pbar.set_description(validate_s)

        avg_train_MSE_loss = total_train_MSE_loss/len(train_dataloader)
        avg_train_MAE_loss = total_train_MAE_loss/len(train_dataloader)
        avg_validate_MSE_loss = total_validate_MSE_loss /len(validate_dataloader)
        avg_validate_MAE_loss = total_validate_MAE_loss /len(validate_dataloader)
        avg_validate_accuracy = total_validate_accuracy /len(validate_dataloader)

        for param_group in optimizer.param_groups:
            record = f"[Epoch: {epoch}/{config.train_epochs}], Train_MSE_loss:{avg_train_MSE_loss:.8f}, Train_MAE_loss:{avg_train_MAE_loss:.8f}, Val_MSE_loss:{avg_validate_MSE_loss:.8f}, Val_RMSE_loss:{math.sqrt(avg_validate_MSE_loss):.8f}, Val_MAE_loss:{avg_validate_MAE_loss:.8f}, Val_accuracy:{avg_validate_accuracy:.8f}, Learning Rate: {param_group['lr']}"
        log_loss(config.path_to_save_loss, config.model_name, record, config.pred_len)
        if (epoch % 1)==0:
            print(record)
        # Record the loss of the training set and validation set to TensorBoard
        # writer.add_scalar("train/train_MSE_loss", avg_train_MSE_loss, epoch)
        # writer.add_scalar("train/train_MAE_loss", avg_train_MAE_loss, epoch)
        # writer.add_scalar("validate/validate_MSE_loss", avg_validate_MSE_loss, epoch)
        # writer.add_scalar("validate/validate_MAE_loss", avg_validate_MAE_loss, epoch)
        # writer.add_scalar("validate/validate_accuracy", avg_validate_accuracy, epoch)
        
        early_stopping(avg_validate_MSE_loss, model, path_to_save_model, config.pred_len, epoch)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        scheduler.step()
    # writer.close()
    return early_stopping.get_best_model() 


def Fine_Tuning_model(config, model, train_dataloader, validate_dataloader, scalar, path_to_save_model, best_pretrain_model, device):
    
    if best_pretrain_model is not None:
        model.load_state_dict(torch.load(path_to_save_model + best_pretrain_model))
    criterion1 = nn.MSELoss()
    criterion2 = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=config.ft_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.step_size, config.ft_gamma)

    # writer = SummaryWriter(f"logs_train_pred{config.pred_len}min；{datetime.now().strftime("%Y-%m-%d；%Hh%Mm%Ss")}/")
    early_stopping = EarlyStopping(patience=config.patience, pretrain=False, verbose=False, delta=0)

    for epoch in range(1, config.train_epochs+1):  
        total_train_MSE_loss = 0
        total_train_MAE_loss = 0
        total_validate_MSE_loss = 0
        total_validate_MAE_loss = 0
        total_validate_accuracy = 0

        model.train()

        pbar = tqdm(train_dataloader)
        for step, (encoder_input, encoder_input_mark, decoder_input ,decoder_input_mark, tgt, _) in enumerate (pbar):

            encoder_input, encoder_input_mark, decoder_input, decoder_input_mark, tgt = encoder_input.to(device), encoder_input_mark.to(device), decoder_input.to(device), decoder_input_mark.to(device), tgt.to(device)

            optimizer.zero_grad()

            # print(encoder_input.shape, encoder_input_mark.shape, decoder_input.shape, decoder_input_mark.shape, tgt.shape)
            output = model(encoder_input, encoder_input_mark, decoder_input, decoder_input_mark)

            train_loss = criterion1(output, tgt)

            train_loss.backward()

            optimizer.step()

            train_output_inverse = output* scalar.std[2] + scalar.mean[2]
            train_target_inverse = tgt* scalar.std[2] + scalar.mean[2]
            train_MSE_loss_inverse = criterion1(train_output_inverse, train_target_inverse)
            train_MAE_loss_inverse = criterion2(train_output_inverse, train_target_inverse)
            total_train_MSE_loss += train_MSE_loss_inverse.item()
            total_train_MAE_loss += train_MAE_loss_inverse.item()

            train_s = "Train ==> [Epoch: {}/{}] - step:{} - train_MSE_loss:{:.8f} - train_MAE_loss:{:.8f}".format(epoch, config.train_epochs, step+1, train_MSE_loss_inverse, train_MAE_loss_inverse)
            pbar.set_description(train_s)

        model.eval()
        with torch.no_grad():

            pbar = tqdm(validate_dataloader)
            for step, (encoder_input, encoder_input_mark, decoder_input ,decoder_input_mark, tgt, _) in enumerate (pbar):

                encoder_input, encoder_input_mark, decoder_input, decoder_input_mark, tgt = encoder_input.to(device), encoder_input_mark.to(device), decoder_input.to(device), decoder_input_mark.to(device), tgt.to(device)

                output = model(encoder_input, encoder_input_mark, decoder_input, decoder_input_mark)

                validate_output_inverse = output* scalar.std[2] + scalar.mean[2]
                validate_target_inverse = tgt* scalar.std[2] + scalar.mean[2]
                validate_MSE_loss_inverse = criterion1(validate_output_inverse, validate_target_inverse)
                validate_MAE_loss_inverse = criterion2(validate_output_inverse, validate_target_inverse)
                total_validate_MSE_loss += validate_MSE_loss_inverse.item()
                total_validate_MAE_loss += validate_MAE_loss_inverse.item()

                correct = (torch.abs(validate_output_inverse - validate_target_inverse) <= config.tolerance).sum().item()
                total_elements = output.numel()  # Total number of elements (16 * 3)
                validate_accuracy = correct / total_elements
                total_validate_accuracy += validate_accuracy

                validate_s = "Validate ==> [Epoch: {}/{}] - step:{} - Val_MSE_loss:{:.8f} - Val_MAE_loss:{:.8f} - Val_accuracy:{:.8f}".format(epoch, config.train_epochs, step+1, validate_MSE_loss_inverse, validate_MAE_loss_inverse, validate_accuracy)
                pbar.set_description(validate_s)

        avg_train_MSE_loss = total_train_MSE_loss/len(train_dataloader)
        avg_train_MAE_loss = total_train_MAE_loss/len(train_dataloader)
        avg_validate_MSE_loss = total_validate_MSE_loss /len(validate_dataloader)
        avg_validate_MAE_loss = total_validate_MAE_loss /len(validate_dataloader)
        avg_validate_accuracy = total_validate_accuracy /len(validate_dataloader)

        for param_group in optimizer.param_groups:
            record = f"[Epoch: {epoch}/{config.train_epochs}], Train_MSE_loss:{avg_train_MSE_loss:.8f}, Train_MAE_loss:{avg_train_MAE_loss:.8f}, Val_MSE_loss:{avg_validate_MSE_loss:.8f}, Val_RMSE_loss:{math.sqrt(avg_validate_MSE_loss):.8f}, Val_MAE_loss:{avg_validate_MAE_loss:.8f}, Val_accuracy:{avg_validate_accuracy:.8f}, Learning Rate: {param_group['lr']}"
        log_loss(config.path_to_save_loss, config.model_name, record, config.pred_len)
        if (epoch % 1)==0:
            print(record)
        # Record the loss of the training set and validation set to TensorBoard
        # writer.add_scalar("train/train_MSE_loss", avg_train_MSE_loss, epoch)
        # writer.add_scalar("train/train_MAE_loss", avg_train_MAE_loss, epoch)
        # writer.add_scalar("validate/validate_MSE_loss", avg_validate_MSE_loss, epoch)
        # writer.add_scalar("validate/validate_MAE_loss", avg_validate_MAE_loss, epoch)
        # writer.add_scalar("validate/validate_accuracy", avg_validate_accuracy, epoch)
        
        early_stopping(avg_validate_MSE_loss, model, path_to_save_model, config.pred_len, epoch)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        scheduler.step()
    # writer.close()
    return early_stopping.get_best_model() 


def prediction(config, model, test_dataloader, scalar, path_to_save_model, best_model, device, Experiment=False):
    
    total_test_MSE_test_loss = 0
    total_test_MAE_test_loss = 0
    total_test_accuracy = 0

    
    model.load_state_dict(torch.load(path_to_save_model + best_model))
    criterion1 = nn.MSELoss()
    criterion2 = nn.L1Loss()
    

    model.eval()
    with torch.no_grad():

        pbar = tqdm(test_dataloader, disable=Experiment)

        for step, (encoder_input, encoder_input_mark, decoder_input ,decoder_input_mark, tgt, _) in enumerate (pbar):

            encoder_input, encoder_input_mark, decoder_input, decoder_input_mark, tgt = encoder_input.to(device), encoder_input_mark.to(device), decoder_input.to(device), decoder_input_mark.to(device), tgt.to(device)

            output = model(encoder_input, encoder_input_mark, decoder_input, decoder_input_mark)

            test_output_inverse = output* scalar.std[2] + scalar.mean[2]
            test_target_inverse = tgt* scalar.std[2] + scalar.mean[2]

            MSE_loss_inverse = criterion1(test_output_inverse,test_target_inverse)
            MAE_loss_inverse = criterion2(test_output_inverse, test_target_inverse)

            total_test_MSE_test_loss += MSE_loss_inverse.item()
            total_test_MAE_test_loss += MAE_loss_inverse.item()

            correct = (torch.abs(test_output_inverse - test_target_inverse) <= config.tolerance).sum().item()
            total_elements = output.numel()  
            test_accuracy = correct / total_elements
            total_test_accuracy += test_accuracy

            s = "test ==> step:{} - MSE_loss:{:.8f} - MAE_loss:{:.8f} - accuracy:{:.8f}".format(step+1, MSE_loss_inverse, MAE_loss_inverse, test_accuracy)
        
            pbar.set_description(s)


    avg_test_MSE_test_loss = total_test_MSE_test_loss /len(test_dataloader)
    avg_test_MAE_test_loss = total_test_MAE_test_loss /len(test_dataloader)
    avg_test_accuracy = total_test_accuracy /len(test_dataloader)

    record = f"Test_MSE_loss:{avg_test_MSE_test_loss:.8f}, Test_RMSE_loss:{math.sqrt(avg_test_MSE_test_loss):.8f}, Test_MAE_loss:{avg_test_MAE_test_loss:.8f}, Test_accuracy:{avg_test_accuracy:.8f}"
    
    if Experiment is False:
        log_loss(config.path_to_save_loss, config.model_name, record, config.pred_len)
    print(record)

    return math.sqrt(avg_test_MSE_test_loss), avg_test_MAE_test_loss


def simple_train_test(config, model, train_dataloader, test_dataloader, device):
    """
    最简单的训练函数：只有训练集训练 + 测试集测试
   
    """
    criterion1 = nn.MSELoss()
    criterion2 = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=config.ft_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.step_size, config.ft_gamma)


    for epoch in range(1, config.train_epochs+1):  
        total_train_MSE_loss = 0
        total_train_MAE_loss = 0

        model.train()

        pbar = tqdm(train_dataloader, disable=True)
        for step, (encoder_input, encoder_input_mark, decoder_input ,decoder_input_mark, tgt, _) in enumerate (pbar):

            encoder_input, encoder_input_mark, decoder_input, decoder_input_mark, tgt = encoder_input.to(device), encoder_input_mark.to(device), decoder_input.to(device), decoder_input_mark.to(device), tgt.to(device)

            optimizer.zero_grad()

            # print(encoder_input.shape, encoder_input_mark.shape, decoder_input.shape, decoder_input_mark.shape, tgt.shape)
            output = model(encoder_input, encoder_input_mark, decoder_input, decoder_input_mark)

            train_loss = criterion1(output, tgt)

            train_loss.backward()

            optimizer.step()

            train_output_inverse = output* train_dataloader.dataset.std[2] +train_dataloader.dataset.mean[2]
            train_target_inverse = tgt* train_dataloader.dataset.std[2] + train_dataloader.dataset.mean[2]
            train_MSE_loss_inverse = criterion1(train_output_inverse, train_target_inverse)
            train_MAE_loss_inverse = criterion2(train_output_inverse, train_target_inverse)
            total_train_MSE_loss += train_MSE_loss_inverse.item()
            total_train_MAE_loss += train_MAE_loss_inverse.item()

            train_s = "Train ==> [Epoch: {}/{}] - step:{} - train_MSE_loss:{:.8f} - train_MAE_loss:{:.8f}".format(epoch, config.train_epochs, step+1, train_MSE_loss_inverse, train_MAE_loss_inverse)
            pbar.set_description(train_s)
        scheduler.step()

        avg_train_MSE_loss = total_train_MSE_loss/len(train_dataloader)
        avg_train_MAE_loss = total_train_MAE_loss/len(train_dataloader)
        record = f"[Epoch: {epoch}/{config.train_epochs}], Train_MSE_loss:{avg_train_MSE_loss:.8f}, Train_MAE_loss:{avg_train_MAE_loss:.8f}"
        print(record)

    model.eval()
    with torch.no_grad():
        total_test_MSE_test_loss = 0
        total_test_MAE_test_loss = 0
        total_test_accuracy = 0
        

        pbar = tqdm(test_dataloader, disable=True)
        
        for step, (encoder_input, encoder_input_mark, decoder_input ,decoder_input_mark, tgt, res) in enumerate (pbar):

            encoder_input, encoder_input_mark, decoder_input, decoder_input_mark, tgt = encoder_input.to(device), encoder_input_mark.to(device), decoder_input.to(device), decoder_input_mark.to(device), tgt.to(device)

            output = model(encoder_input, encoder_input_mark, decoder_input, decoder_input_mark)

            test_output_inverse = output* train_dataloader.dataset.std[2] + train_dataloader.dataset.mean[2]
            test_target_inverse = tgt* train_dataloader.dataset.std[2] + train_dataloader.dataset.mean[2]

            MSE_loss_inverse = criterion1(test_output_inverse,test_target_inverse)
            MAE_loss_inverse = criterion2(test_output_inverse, test_target_inverse)

            total_test_MSE_test_loss += MSE_loss_inverse.item()
            total_test_MAE_test_loss += MAE_loss_inverse.item()

            correct = (torch.abs(test_output_inverse - test_target_inverse) <= config.tolerance).sum().item()
            total_elements = output.numel()  
            test_accuracy = correct / total_elements
            total_test_accuracy += test_accuracy

            s = "test ==> step:{} - MSE_loss:{:.8f} - MAE_loss:{:.8f} - accuracy:{:.8f}".format(step+1, MSE_loss_inverse, MAE_loss_inverse, test_accuracy)
            pbar.set_description(s)

    avg_test_MSE_test_loss = total_test_MSE_test_loss /len(test_dataloader)
    avg_test_MAE_test_loss = total_test_MAE_test_loss /len(test_dataloader)
    avg_test_accuracy = total_test_accuracy /len(test_dataloader)

    record = f"Test_MSE_loss:{avg_test_MSE_test_loss:.8f}, Test_RMSE_loss:{math.sqrt(avg_test_MSE_test_loss):.8f}, Test_MAE_loss:{avg_test_MAE_test_loss:.8f}, Test_accuracy:{avg_test_accuracy:.8f}"
    # log_loss(config.path_to_save_loss, config.model_name, record, config.pred_len)
    print(record)

    return math.sqrt(avg_test_MSE_test_loss), avg_test_MAE_test_loss


def inference_individual(config, model, test_dataloader, scalar, path_to_save_model, best_model, device, patient):
    
    total_test_MSE_test_loss = 0
    total_test_MAE_test_loss = 0
    total_test_accuracy = 0

    
    model.load_state_dict(torch.load(path_to_save_model + best_model))
    criterion1 = nn.MSELoss()
    criterion2 = nn.L1Loss()
    

    model.eval()
    with torch.no_grad():

        pbar = tqdm(test_dataloader)
        all_true_values = [] 
        all_predicted_values = []  
        for step, (encoder_input, encoder_input_mark, decoder_input ,decoder_input_mark, tgt, _) in enumerate (pbar):

            encoder_input, encoder_input_mark, decoder_input, decoder_input_mark, tgt = encoder_input.to(device), encoder_input_mark.to(device), decoder_input.to(device), decoder_input_mark.to(device), tgt.to(device)

            output = model(encoder_input, encoder_input_mark, decoder_input, decoder_input_mark)

            test_output_inverse = output* scalar.std[2] + scalar.mean[2]
            test_target_inverse = tgt* scalar.std[2] + scalar.mean[2]

            MSE_loss_inverse = criterion1(test_output_inverse,test_target_inverse)
            MAE_loss_inverse = criterion2(test_output_inverse, test_target_inverse)

            total_test_MSE_test_loss += MSE_loss_inverse.item()
            total_test_MAE_test_loss += MAE_loss_inverse.item()

            correct = (torch.abs(test_output_inverse - test_target_inverse) <= config.tolerance).sum().item()
            total_elements = output.numel()  
            test_accuracy = correct / total_elements
            total_test_accuracy += test_accuracy

            #可视化预测效果
            # 保证 append 的是 numpy 数组，避免 tensor 直接拼接报错
            all_true_values.append(test_target_inverse.detach().cpu().numpy())
            all_predicted_values.append(test_output_inverse.detach().cpu().numpy())

            s = "test ==> step:{} - MSE_loss:{:.8f} - MAE_loss:{:.8f} - accuracy:{:.8f}".format(step+1, MSE_loss_inverse, MAE_loss_inverse, test_accuracy)
            pbar.set_description(s)
    
    path_to_save_prediction = f"save_{config.model_name}_prediction_{config.seed}seed_{config.pred_len}min_{patient}patient/"
    if os.path.exists(path_to_save_prediction):
        shutil.rmtree(path_to_save_prediction)
    os.makedirs(path_to_save_prediction)
    all_true_values_np = np.concatenate(all_true_values, axis=0)  # (total_samples, time_steps, 1)
    all_predicted_values_np = np.concatenate(all_predicted_values, axis=0)  # (total_samples, time_steps, 1)
    np.save(os.path.join(path_to_save_prediction, f"true_values.npy"), all_true_values_np)
    np.save(os.path.join(path_to_save_prediction, f"predicted_values.npy"), all_predicted_values_np)

    avg_test_MSE_test_loss = total_test_MSE_test_loss /len(test_dataloader)
    avg_test_MAE_test_loss = total_test_MAE_test_loss /len(test_dataloader)
    avg_test_accuracy = total_test_accuracy /len(test_dataloader)

    record = f"Test_MSE_loss:{avg_test_MSE_test_loss:.8f}, Test_RMSE_loss:{math.sqrt(avg_test_MSE_test_loss):.8f}, Test_MAE_loss:{avg_test_MAE_test_loss:.8f}, Test_accuracy:{avg_test_accuracy:.8f}"
    print(record)

    return math.sqrt(avg_test_MSE_test_loss), avg_test_MAE_test_loss

def save_inference_individual_result(config, path_to_save_model, best_model, model=None):

    time_step = config.time_step
    seq_len = int(config.seq_len / time_step)
    label_len = int(config.label_len / time_step)
    pred_len = int(config.pred_len / time_step)

    if hasattr(config, 'seg_len') and config.seg_len is not None:
        seg_len = int(config.seg_len / time_step)
    elif hasattr(config, 'patch_len') and config.patch_len is not None:
        seg_len = int(config.patch_len / time_step)
    else:
        seg_len = None 


    device = torch.device(config.device)

    data_dir="../Glucose_Data/OhioT1DM_processed_dataset"
    _, _, _, scalar = prepare_Ohio_data(
        data_dir=data_dir,
        seq_length=seq_len, label_length=label_len, pred_length=pred_len)
    patients = ["563", "596"]
    for p in patients:
        test_dataset = torch.utils.data.ConcatDataset([
            OhioDataset(
                raw_df=pd.read_csv(os.path.join(data_dir, "{}_train.csv".format(p))), 
                seq_length=seq_len, label_length=label_len, pred_length=pred_len, 
                external_mean=scalar.mean,
                external_std=scalar.std,    
            ),
            OhioDataset(
                raw_df=pd.read_csv(os.path.join(data_dir, "{}_test.csv".format(p))), 
                seq_length=seq_len, label_length=label_len, pred_length=pred_len, 
                external_mean=scalar.mean,
                external_std=scalar.std,
            )
        ])

        test_dataloader = DataLoader(test_dataset, config.batch_size, shuffle=False)

        model = model.to(device)
        
        print(f"----------------[Inference] Starting individualized prediction for subject {p}----------------")

        inference_individual(config, model, test_dataloader, scalar, path_to_save_model, best_model, device, patient=p)
        print(f"----------------[Inference] Finished evaluation on test dataset for subject {p}----------------")


def run(config, model, Tranning_mode="normal_train", Experiment_mode=None):
    """
    mode: 
        "pretrain"                - Pre-training + Fine-tuning + Prediction
        "finetune"&"normal_train" - Fine-tuning/Normal training + Prediction only
        "predict"                 - Prediction only
    """
    Experiment_mode_dict = {
            "ggg": "Glucose-Glucose-Glucose",
            "cgg": "Carbs-Glucose-Glucose",
            "gig": "Glucose-Insulin-Glucose",
        }
    if Experiment_mode is not None:
        results_dir = f"feature_fusion_{config.model_name}_results/"
        config.path_to_save_loss = f"{results_dir}/save_loss/"
        config.path_to_save_scaler = f"{results_dir}/save_scaler/"
        path_to_save_model = f"{results_dir}/save_{config.model_name}_model_{config.seed}seed_{config.pred_len}min/"
    else:
        path_to_save_model = f"save_{config.model_name}_model_{config.seed}seed_{config.pred_len}min/"

    time_step = config.time_step
    seq_len = int(config.seq_len/time_step)
    label_len = int(config.label_len/time_step)
    pred_len = int(config.pred_len/time_step)
    device = torch.device(config.device)

    train_dataset, validate_dataset, test_dataset, scalar = prepare_Ohio_data(
        data_dir="../Glucose_Data/OhioT1DM_processed_dataset",
        seq_length=seq_len, label_length=label_len, pred_length=pred_len, Experiment_mode=Experiment_mode)
    if Experiment_mode is not None  and config.enc_in == 1:
        train_dataloader = DataLoader(SingleFeatureWrapper(train_dataset), batch_size=config.batch_size, shuffle=True)
        validate_dataloader = DataLoader(SingleFeatureWrapper(validate_dataset), batch_size=config.batch_size, shuffle=False)
        test_dataloader = DataLoader(SingleFeatureWrapper(test_dataset), batch_size=config.batch_size, shuffle=False)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        validate_dataloader = DataLoader(validate_dataset, batch_size=config.batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    model = model.to(device)

    if Tranning_mode == "pretrain":
        if Experiment_mode is not None:
            os.makedirs(results_dir, exist_ok=True)
            os.makedirs(config.path_to_save_scaler, exist_ok=True)
            os.makedirs(path_to_save_model, exist_ok=True)
            print(f"\n========================Experiment mode: {Experiment_mode_dict.get(Experiment_mode)}========================")
            log_loss(config.path_to_save_loss, config.model_name, f"========================Experiment mode: {Experiment_mode_dict.get(Experiment_mode)}========================", config.pred_len)            
        else:
            clean_directory(path_to_save_model)
        pretrain_dataset = T1DMS_GlucoseDataset(
            sim_result_mat_file_name='sim_results_train_dataset.mat',
            sim_data_mat_file_name='sim_data_train_dataset.mat',
            root_dir='../Glucose_Data/T1DMS_GlucoseDataset', train_dataset=True, Experiment_mode=Experiment_mode,
            seq_length=seq_len, label_length=label_len, pred_length=pred_len,
            stride=config.stride, sensor_sampling=time_step, path_to_save_scaler=config.path_to_save_scaler)
        prevalidate_dataset = T1DMS_GlucoseDataset(
            sim_result_mat_file_name='sim_results_validate_dataset.mat',
            sim_data_mat_file_name='sim_data_validate_dataset.mat',
            root_dir='../Glucose_Data/T1DMS_GlucoseDataset', train_dataset=False, Experiment_mode=Experiment_mode,
            seq_length=seq_len, label_length=label_len, pred_length=pred_len,
            stride=config.stride, sensor_sampling=time_step, path_to_save_scaler=config.path_to_save_scaler)
        if Experiment_mode is not None and config.enc_in == 1:
            pretrain_dataloader = DataLoader(SingleFeatureWrapper(pretrain_dataset), config.batch_size, shuffle=True)
            prevalidate_dataloader = DataLoader(SingleFeatureWrapper(prevalidate_dataset), config.batch_size, shuffle=False)
        else:
            pretrain_dataloader = DataLoader(pretrain_dataset, config.batch_size, shuffle=True)
            prevalidate_dataloader = DataLoader(prevalidate_dataset, config.batch_size, shuffle=False)

        print(f"-------------{config.model_name} pre-training starts-------------")
        log_loss(config.path_to_save_loss, config.model_name, f"-------------{config.model_name} pre-training starts-------------", config.pred_len)
        print("Options =================>")
        print(vars(config))
        log_loss(config.path_to_save_loss, config.model_name, str(vars(config)), config.pred_len)
        best_pretrain_model = pretrain_model(config, model, pretrain_dataloader, prevalidate_dataloader, path_to_save_model, device)

        print(f"-------------{config.model_name} Pre-training complete, fine-tuning begins-------------")
        log_loss(config.path_to_save_loss, config.model_name, f"-------------{config.model_name} Pre-training complete, fine-tuning begins-------------", config.pred_len)
        print(f"best_pretrain_model:{best_pretrain_model}")
        log_loss(config.path_to_save_loss, config.model_name, best_pretrain_model, config.pred_len)
        best_model = Fine_Tuning_model(config, model, train_dataloader, validate_dataloader, scalar, path_to_save_model, best_pretrain_model, device)

    elif Tranning_mode in ["finetune", "normal_train"]:

        best_pretrain_model = getattr(config, "best_pretrain_model", None)
        if Tranning_mode == "finetune":
            # Fine-tuning mode: must have pretrain weights
            if best_pretrain_model is None:
                raise ValueError("Fine-tuning mode requires a pre-trained model! Please set config.best_pretrain_model.")
            print(f"best_pretrain_model:{best_pretrain_model}")
            record = f"-------------{config.model_name} fine tuning begins-------------"
        else:
            if Experiment_mode is not None:
                os.makedirs(results_dir, exist_ok=True)
                os.makedirs(config.path_to_save_scaler, exist_ok=True)
                os.makedirs(path_to_save_model, exist_ok=True)
                print(f"\n========================Experiment mode: {Experiment_mode_dict.get(Experiment_mode)}========================")
                log_loss(config.path_to_save_loss, config.model_name, f"========================Experiment mode: {Experiment_mode_dict.get(Experiment_mode)}========================", config.pred_len)
            else:
                clean_directory(path_to_save_model)
            record = f"-------------{config.model_name} normal training starts-------------"
        print(record)
        log_loss(config.path_to_save_loss, config.model_name, record, config.pred_len)
        print("Options =================>")
        print(vars(config))
        log_loss(config.path_to_save_loss, config.model_name, str(vars(config)), config.pred_len)
        best_model = Fine_Tuning_model(config, model, train_dataloader, validate_dataloader, scalar, path_to_save_model, best_pretrain_model, device)

    elif Tranning_mode == "predict":

        best_model = config.best_model
        if best_model is None:
            raise ValueError("config.best_model is None! Please specify the model path or name for prediction.")

    print(f"-------------{config.model_name} prediction starts-------------")
    log_loss(config.path_to_save_loss, config.model_name, f"-------------{config.model_name} prediction starts-------------", config.pred_len)
    if Tranning_mode == "predict":
        print("Options =================>")
        print(vars(config))
        log_loss(config.path_to_save_loss, config.model_name, str(vars(config)), config.pred_len)
    print(f"best_model:{best_model}")
    log_loss(config.path_to_save_loss, config.model_name, best_model, config.pred_len)
    RMSE, MAE = prediction(config, model, test_dataloader, scalar, path_to_save_model, best_model, device)
    
    if Experiment_mode is None:
        if config.save_pred:
            print(f"-------------{config.model_name} prediction complete, inference individual begins-------------")
            save_inference_individual_result(config, path_to_save_model, best_model, model)

    return RMSE, MAE


def inference_and_save_attention(config, model, test_dataloader, path_to_save_model, best_model, device, patient):
    """
    A core function dedicated to reasoning and preserving attention scores.
    """
    # Load the best model weights
    model.load_state_dict(torch.load(path_to_save_model + best_model))
    
    # Set the model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        pbar = tqdm(test_dataloader)
        pbar.set_description(f"Saving Attention for Patient {patient}")
        
        all_attn_weights = []

        for step, (encoder_input, encoder_input_mark, decoder_input, decoder_input_mark, tgt, _) in enumerate(pbar):
            # Move data to the device
            encoder_input, encoder_input_mark, decoder_input, decoder_input_mark, tgt = encoder_input.to(device), encoder_input_mark.to(device), decoder_input.to(device), decoder_input_mark.to(device), tgt.to(device)

            # Forward pass and get attention weights
            # Assume the model supports return_attention=True parameter
            _, attn_weights = model(encoder_input, encoder_input_mark, decoder_input, decoder_input_mark)
            
            # Collect attention weights
            all_attn_weights.append(attn_weights.detach().cpu().numpy())

    # Create save directory
    path_to_save_attention = f"save_{config.model_name}_attention_{config.seed}seed_{config.pred_len}min_{patient}patient/"
    if os.path.exists(path_to_save_attention):
        shutil.rmtree(path_to_save_attention)
    os.makedirs(path_to_save_attention)

    all_attn_weights_np = np.concatenate(all_attn_weights, axis=0)
    np.save(os.path.join(path_to_save_attention, "attention_weights.npy"), all_attn_weights_np)
    
    print(f"Attention maps saved to {path_to_save_attention}")
    print(f"Saved attention shape: {all_attn_weights_np.shape}")


def run_attention_saving_experiment(config, path_to_save_model, best_model, model=None):

    time_step = config.time_step
    seq_len = int(config.seq_len / time_step)
    label_len = int(config.label_len / time_step)
    pred_len = int(config.pred_len / time_step)

    device = torch.device(config.device)

    data_dir="../Glucose_Data/OhioT1DM_processed_dataset"

    _, _, _, scalar = prepare_Ohio_data(
        data_dir=data_dir,
        seq_length=seq_len, label_length=label_len, pred_length=pred_len)
    
    patients_to_analyze = ["563", "596"]
    
    for p in patients_to_analyze:
        test_dataset = torch.utils.data.ConcatDataset([
            OhioDataset(
                raw_df=pd.read_csv(os.path.join(data_dir, f"{p}_train.csv")), 
                seq_length=seq_len, label_length=label_len, pred_length=pred_len, 
                external_mean=scalar.mean,
                external_std=scalar.std,    
            ),
            OhioDataset(
                raw_df=pd.read_csv(os.path.join(data_dir, f"{p}_test.csv")), 
                seq_length=seq_len, label_length=label_len, pred_length=pred_len, 
                external_mean=scalar.mean,
                external_std=scalar.std,
            )
        ])

        test_dataloader = DataLoader(test_dataset, config.batch_size, shuffle=False)

        model = model.to(device)
        
        print(f"----------------[Attention] Starting to save attention maps for subject {p}----------------")
        
        inference_and_save_attention(
            config, 
            model, 
            test_dataloader, 
            path_to_save_model, 
            best_model, 
            device, 
            patient=p
        )
        
        print(f"----------------[Attention] Finished saving attention maps for subject {p}----------------")






