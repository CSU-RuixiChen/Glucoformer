import sys
import os
ROOT_DIR = os.path.abspath(os.path.join(os.getcwd(), "../"))
sys.path.insert(0, ROOT_DIR)
from Glucose_Data.T1DMS_GlucoseDataset import T1DMS_GlucoseDataset
from Glucose_Data.OhioDataset import *
from Glucoformer.Glucoformer_models.Glucoformer import *
from Glucoformer.utils.train_eval import *
from Glucoformer.utils.tools import *
import random
import numpy as np
import pandas as pd

def personalized_prediction(config, patient_id):

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    time_step = config.time_step
    seq_len = int(config.seq_len/time_step)
    label_len = int(config.label_len/time_step)
    pred_len = int(config.pred_len/time_step)
    seg_len = int(config.seg_len/time_step)

    device = torch.device(config.device)

    patients = ["563", "596"]
    p = patients[patient_id]

    print("----------------Start training dataset for subject {}-------------".format(p))
    train_dataset, test_dataset = prepare_personal_data(os.path.join("../Glucose_Data/OhioT1DM_processed_dataset", f"{p}_train.csv"), 
                                                        os.path.join("../Glucose_Data/OhioT1DM_processed_dataset", f"{p}_test.csv"), 
                                                        seq_length=seq_len, label_length=label_len, pred_length=pred_len)

    train_dataloader = DataLoader(train_dataset, config.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, config.batch_size, shuffle=False)

    model = Glucoformer(data_dim = config.data_dim, in_len = seq_len, out_len = pred_len, seg_len = seg_len,  output_size = config.c_out,
                    factor=config.factor, d_model=config.d_model, d_ff=config.d_ff, n_heads=config.n_heads, e_layers=config.e_layers, dropout=config.dropout).to(device)

    RMSE, MAE = simple_train_test(config, model, train_dataloader, test_dataloader, device)

    print("----------------Testing dataset for subject {}-------------".format(p))

    return RMSE, MAE


def generalized_prediction(config, patient_id=0):
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    time_step = config.time_step
    seq_len = int(config.seq_len / time_step)
    label_len = int(config.label_len / time_step)
    pred_len = int(config.pred_len / time_step)
    seg_len = int(config.seg_len / time_step)

    path_to_save_model = f"../Glucoformer/save_{config.model_name}_model_{config.seed}seed_{config.pred_len}min/"
    device = torch.device(config.device)

    _, _, _, scalar = prepare_Ohio_data(
        data_dir="../Glucose_Data/OhioT1DM_processed_dataset",
        seq_length=seq_len, label_length=label_len, pred_length=pred_len)
    patients = ["563", "596"]
    p = patients[patient_id]
    print("----------------Start training dataset for subject {}-------------".format(p))
    test_df = pd.read_csv(os.path.join("../Glucose_Data/OhioT1DM_processed_dataset", f"{p}_test.csv"))

    test_dataset = OhioDataset(
        test_df, seq_len, label_len, pred_len,
        external_mean=scalar.mean, external_std=scalar.std)

    test_dataloader = DataLoader(test_dataset, config.batch_size, shuffle=False)

    model = Glucoformer(
        data_dim=config.data_dim, in_len=seq_len, out_len=pred_len, seg_len=seg_len, output_size=config.c_out,
        factor=config.factor, d_model=config.d_model, d_ff=config.d_ff, n_heads=config.n_heads,
        e_layers=config.e_layers, dropout=config.dropout
    ).to(device)


    best_model = config.best_model

    RMSE, MAE = prediction(config, model, test_dataloader, scalar, path_to_save_model, best_model, device, Experiment=True)

    print("----------------Complete Testing dataset for subject {}-------------".format(p))

    return RMSE, MAE


def plot_figure(act_sig, pred_sig, model_name, PH, patient, path=None):
    plt.figure(figsize=(10, 3.5))
    plt.plot(act_sig, label='Actual')
    plt.plot(pred_sig, linestyle='-', label='Predicted')
    # plt.xticks(test_times[::5], rotation=45)
    plt.xlabel('Time')
    plt.ylabel('Blood Glucose')
    plt.title(f'Blood Glucose Prediction ({model_name}, PH={PH}min, Patient {patient})')
    plt.legend()
    plt.show()
    if path is not None:
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        plt.savefig(path, bbox_inches='tight')


def Prediction_Visualiser(PH, seed=2024, days=3, start_time=0):
    Duration=int(288*days)
    patients = [
        "563", 
                # "596"
                ]
    model_names = [
        "Glucoformer", 
        "Crossformer", 
        "PatchTST", 
        "TimeXer",
        "DLinear", 
        "Informer", 
        "Transformer", 
        "LSTM", 
        "GRU"
    ]
    for patient in patients:
        for model_name in model_names:
            y_path = f'../{model_name}/save_{model_name}_prediction_{seed}seed_{PH}min_{patient}patient/true_values.npy'
            yp_path = f'../{model_name}/save_{model_name}_prediction_{seed}seed_{PH}min_{patient}patient/predicted_values.npy'
            y = np.load(y_path)[start_time : Duration+start_time, -1, 0]
            yp = np.load(yp_path)[start_time : Duration+start_time, -1, 0]
            plot_figure(y, yp, model_name, PH, patient=patient)
        print(f"PH:{PH}, patient:{patient}", np.load(y_path).shape, np.load(yp_path).shape)
    

def plot_multi_figure(PH, seed=2024, patient=563, days=3, start_time=0):
    Duration = int(288 * days)
    model_names = [
        "GRU",
        "LSTM", 
        "Transformer", 
        "Informer", 
        "DLinear", 
        "TimeXer",
        "PatchTST", 
        "Crossformer", 
        "Glucoformer"
    ]

    model_colors = {
        "Glucoformer":   "#E41A1C",  # Dark Red
        "Crossformer":   "#377EB8",  # Dark Blue
        "PatchTST":      "#228B22",  # Forest Green
        "TimeXer":       "#984EA3",  # Purple
        "DLinear":       "#FF7F00",  # Orange
        "Informer":      "#A65628",  # Brown
        "Transformer":   "#F781BF",  # Magenta
        "LSTM":          "#17BECF",  # Cyan
        "GRU":           "#72777b",  
    }

    fig, ax = plt.subplots(figsize=(10, 3.5))

    y_path = f'../Glucoformer/save_Glucoformer_prediction_{seed}seed_{PH}min_{patient}patient/true_values.npy'
    y = np.load(y_path)[start_time:Duration+start_time, -1, 0]
    plt.plot(y, label='Actual', color='k', linewidth=1.8)
    
    for model_name in model_names:
        yp_path = f'../{model_name}/save_{model_name}_prediction_{seed}seed_{PH}min_{patient}patient/predicted_values.npy'
        if not os.path.exists(yp_path):
            continue
        yp = np.load(yp_path)[start_time:Duration+start_time, -1, 0]
        if model_name == "Glucoformer":
            plt.plot(yp, label='Glucoformer', color=model_colors[model_name], linewidth=1)
        else:
            plt.plot(yp, label=model_name, linestyle='--', linewidth=1, alpha=1,
                        color=model_colors.get(model_name, "gray"))

    plt.xlabel('Time')
    plt.ylabel('Blood Glucose')
    plt.title(f'Blood Glucose Prediction Comparison (PH={PH}min, Patient {patient})')
    
    plt.legend(
        loc='best',      
        ncol=5,         
        fontsize=10,
        frameon=True,
        edgecolor="gray",
        facecolor="white",  
        framealpha=0.7      
    )
    
    plt.tight_layout()
    plt.show()


def generate_main_and_zoom_plots(PH, seed, patient, days, start_time, zoom_regions, base_zoom_height=5):
    import matplotlib.patches as patches
    """
    Generates a main prediction plot with zoom indicators, and then creates separate plots for each zoom region.

    Args:
        PH (int): Prediction horizon in minutes.
        seed (int): Random seed for the experiment.
        patient (int or str): Patient identifier.
        days (float): Duration of the plot in days.
        start_time (int): Starting index for the data slice.
        zoom_regions (list): A list of dictionaries, each defining a zoom region with 'x_range' and 'y_range'.
        base_zoom_height (float): The base height in inches for the separate zoom plots.
    """
    # --- Step 1: Generate the main plot ---
    Duration = int(288 * days)
    model_names = ["Transformer", "PatchTST", "Crossformer", "Glucoformer"]
    model_colors = {
        "Glucoformer":   "#E41A1C",  # Red
        "Crossformer":   "#377EB8",  # Blue
        "PatchTST":      "#228B22",  # Forest Green
        "Transformer":   "#F781BF",  # Pink
    }

    fig, ax = plt.subplots(figsize=(10, 3.25))

    # Plot ground truth
    y_path = f'../Glucoformer/save_Glucoformer_prediction_{seed}seed_{PH}min_{patient}patient/true_values.npy'
    y = np.load(y_path)[start_time:Duration+start_time, -1, 0]
    ax.plot(y, label='Actual', color='k', linewidth=1.8)
    
    # Plot model predictions
    for model_name in model_names:
        yp_path = f'../{model_name}/save_{model_name}_prediction_{seed}seed_{PH}min_{patient}patient/predicted_values.npy'
        if os.path.exists(yp_path):
            yp = np.load(yp_path)[start_time:Duration+start_time, -1, 0]
            linestyle = '-' if model_name == "Glucoformer" else '--'
            linewidth = 1.2 if model_name == "Glucoformer" else 1.0
            ax.plot(yp, label=model_name, linestyle=linestyle, linewidth=linewidth, color=model_colors.get(model_name, "gray"))

    # Style the main plot
    ax.set_xlabel('Time')
    ax.set_ylabel('Blood Glucose (mg/dL)')
    ax.set_title(f'Prediction Comparison (PH={PH}min, Patient {patient})')
    ax.legend(loc='best', ncol=5, fontsize=9)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15)

    # --- Step 2: Draw indicator boxes on the main plot ---
    for region in zoom_regions:
        rect = patches.Rectangle(
            (region['x_range'][0], region['y_range'][0]),
            width=region['x_range'][1] - region['x_range'][0],
            height=region['y_range'][1] - region['y_range'][0],
            linewidth=1.2, edgecolor='black', facecolor='none', alpha=0.8
        )
        ax.add_patch(rect)
    
    print("Main plot with indicator boxes has been generated.")
    plt.show()

    # --- Step 3: Generate and show a separate plot for each zoom region ---
    print("\nGenerating individual plots for each zoom region...")
    for region in zoom_regions:
        x_range, y_range = region['x_range'], region['y_range']
        
        # Calculate aspect ratio
        data_width = x_range[1] - x_range[0]
        data_height = y_range[1] - y_range[0]
        fig_width = base_zoom_height * (data_width / data_height) if data_height > 0 else base_zoom_height

        # Create new figure for the zoom plot
        fig_zoom, ax_zoom = plt.subplots(figsize=(fig_width, base_zoom_height))

        # Re-plot all lines
        for line in ax.get_lines():
            ax_zoom.plot(line.get_xdata(), line.get_ydata(),
                         color=line.get_color(),
                         linestyle=line.get_linestyle(),
                         linewidth=line.get_linewidth() * 1.5)

        # Set limits and style the zoom plot
        ax_zoom.set_xlim(x_range)
        ax_zoom.set_ylim(y_range)
        ax_zoom.grid(True, linestyle='--', alpha=0.7)
        ax_zoom.set_title("")
        ax_zoom.set_xlabel("")
        ax_zoom.set_ylabel("")
        if ax_zoom.get_legend() is not None:
            ax_zoom.get_legend().remove()

        plt.tight_layout()
        plt.show()
        

def clarke(y, yp, model_name, PH, patient, print_figure=False):
    """
    Performs Clarke Error Grid Analysis
    
    The Clarke error grid approach is used to assess the clinical
    significance of differences between the glucose measurement technique
    under test and the venous blood glucose reference measurements.
    
    Parameters:
    -----------
    y : array_like
        Reference values (mg/dl)
    yp : array_like
        Predicted/estimated values (mg/dl)
    print_figure : bool, optional
        Whether to save the figure (default is True)
    
    Returns:
    --------
    total : ndarray
        Total points per zone: 
        total[0] = zone A, 
        total[1] = zone B, and so on
    percentage : ndarray
        Percentage of data which fell in certain region:
        percentage[0] = zone A, 
        percentage[1] = zone B, and so on.
    """
    
    # Error checking
    if len(y) == 0 or len(yp) == 0:
        raise ValueError('There are no inputs.')
    
    if len(yp) != len(y):
        raise ValueError('Vectors y and yp must be the same length.')
    
    # Determine data length
    n = len(y)
    
    # Plot Clarke's Error Grid
    plt.figure(figsize=(6, 6))
    plt.plot(y, yp, 'ko', markersize=1, markerfacecolor='k', markeredgecolor='k', alpha=0.35)
    plt.xlabel('Reference Concentration [mg/dl]')
    plt.ylabel('Predicted Concentration [mg/dl]')
    plt.title(f"Clarke's Error Grid Analysis\n({model_name}, PH={PH}min, Patient {patient})")
    plt.xlim([0, 400])
    plt.ylim([0, 400])
    plt.gca().set_aspect('equal')
    
    # Theoretical 45° regression line
    plt.plot([0, 400], [0, 400], 'k:')
    
    # Zone boundaries
    plt.plot([0, 175/3], [70, 70], 'k-')
    plt.plot([175/3, 400/1.2], [70, 400], 'k-')
    plt.plot([70, 70], [84, 400], 'k-')
    plt.plot([0, 70], [180, 180], 'k-')
    plt.plot([70, 290], [180, 400], 'k-')
    plt.plot([70, 70], [0, 56], 'k-')
    plt.plot([70, 400], [56, 320], 'k-')
    plt.plot([180, 180], [0, 70], 'k-')
    plt.plot([180, 400], [70, 70], 'k-')
    plt.plot([240, 240], [70, 180], 'k-')
    plt.plot([240, 400], [180, 180], 'k-')
    plt.plot([130, 180], [0, 70], 'k-')
    
    # Zone labels
    plt.text(30, 20, 'A', fontsize=12)
    plt.text(30, 150, 'D', fontsize=12)
    plt.text(30, 380, 'E', fontsize=12)
    plt.text(150, 380, 'C', fontsize=12)
    plt.text(160, 20, 'C', fontsize=12)
    plt.text(380, 20, 'E', fontsize=12)
    plt.text(380, 120, 'D', fontsize=12)
    plt.text(380, 260, 'B', fontsize=12)
    plt.text(280, 380, 'B', fontsize=12)
    
    if print_figure:
        plt.savefig('Clarke_EGA.png', dpi=300, bbox_inches='tight')
        # plt.savefig('Clarke_EGA.emf')  # EMF format not directly supported in matplotlib
    
    # Initialize output
    total = np.zeros(5)
    
    # Statistics
    for i in range(n):
        if (yp[i] <= 70 and y[i] <= 70) or (yp[i] <= 1.2*y[i] and yp[i] >= 0.8*y[i]):
            total[0] += 1  # Zone A
        else:
            if ((y[i] >= 180) and (yp[i] <= 70)) or ((y[i] <= 70) and yp[i] >= 180):
                total[4] += 1  # Zone E
            else:
                if ((y[i] >= 70 and y[i] <= 290) and (yp[i] >= y[i] + 110)) or ((y[i] >= 130 and y[i] <= 180) and (yp[i] <= (7/5)*y[i] - 182)):
                    total[2] += 1  # Zone C
                else:
                    if ((y[i] >= 240) and ((yp[i] >= 70) and (yp[i] <= 180))) or (y[i] <= 175/3 and (yp[i] <= 180) and (yp[i] >= 70)) or ((y[i] >= 175/3 and y[i] <= 70) and (yp[i] >= (6/5)*y[i])):
                        total[3] += 1  # Zone D
                    else:
                        total[1] += 1  # Zone B
    
    percentage = (total / n) * 100
    
    return total, percentage


def clarke_colored_zones(y, yp, model_name, PH, patient, ax=None, print_figure=False, draw=True):
    """
    Performs Clarke Error Grid Analysis

    Parameters:
    -----------
    y : array_like
        Reference values (mg/dl)
    yp : array_like
        Predicted/estimated values (mg/dl)
    ax : matplotlib.axes.Axes or None
        Subplot object, if None, a new one is created automatically
    print_figure : bool, optional
        Whether to save the figure

    Returns:
    --------
    total : ndarray
        Total points per zone: total[0]=A, total[1]=B, ...
    percentage : ndarray
        Percentage of data in each zone
    """
    if len(y) == 0 or len(yp) == 0:
        raise ValueError('There are no inputs.')
    if len(yp) != len(y):
        raise ValueError('Vectors y and yp must be the same length.')

    n = len(y)
    points_A, points_B, points_other = [], [], []
    total = np.zeros(5)

    for i in range(n):
        if (yp[i] <= 70 and y[i] <= 70) or (yp[i] <= 1.2*y[i] and yp[i] >= 0.8*y[i]):
            total[0] += 1  # Zone A
            points_A.append((y[i], yp[i]))
        else:
            if ((y[i] >= 180) and (yp[i] <= 70)) or ((y[i] <= 70) and yp[i] >= 180):
                total[4] += 1  # Zone E
                points_other.append((y[i], yp[i]))
            else:
                if ((y[i] >= 70 and y[i] <= 290) and (yp[i] >= y[i] + 110)) or ((y[i] >= 130 and y[i] <= 180) and (yp[i] <= (7/5)*y[i] - 182)):
                    total[2] += 1  # Zone C
                    points_other.append((y[i], yp[i]))
                else:
                    if ((y[i] >= 240) and ((yp[i] >= 70) and (yp[i] <= 180))) or (y[i] <= 175/3 and (yp[i] <= 180) and (yp[i] >= 70)) or ((y[i] >= 175/3 and y[i] <= 70) and (yp[i] >= (6/5)*y[i])):
                        total[3] += 1  # Zone D
                        points_other.append((y[i], yp[i]))
                    else:
                        total[1] += 1  # Zone B
                        points_B.append((y[i], yp[i]))
    if draw:
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        # if points_A:
        #     ax.scatter(*zip(*points_A), c='k', s=1, alpha=0.35, label='Zone A')
        # if points_B:
        #     ax.scatter(*zip(*points_B), c='b', s=1, alpha=0.35, label='Zone B')
        # if points_other:
        #     ax.scatter(*zip(*points_other), c='r', s=1, alpha=0.35, label='Other Zones')
        
        if points_A:
            ax.scatter(*zip(*points_A), c='#2ca02c', s=1, alpha=0.35, label='Zone A')      # 绿色
        if points_B:
            ax.scatter(*zip(*points_B), c='#1f77b4', s=1, alpha=0.35, label='Zone B')      # 蓝色
        if points_other:
            ax.scatter(*zip(*points_other), c='#d62728', s=1, alpha=0.35, label='Other Zones')  # 红色

        ax.set_xlabel('Reference Concentration [mg/dl]')
        ax.set_ylabel('Predicted Concentration [mg/dl]')
        if ax is None:
            ax.set_title(f"Clarke's Error Grid\n({model_name}, PH={PH}min, Patient {patient})", fontsize=10)
        else:
            ax.set_title(f"{model_name}", fontsize=13)
        ax.set_xlim([0, 400])
        ax.set_ylim([0, 400])
        ax.set_aspect('equal')

        # 45° line
        ax.plot([0, 400], [0, 400], 'k:')

        # Zone boundaries
        ax.plot([0, 175/3], [70, 70], 'k-')
        ax.plot([175/3, 400/1.2], [70, 400], 'k-')
        ax.plot([70, 70], [84, 400], 'k-')
        ax.plot([0, 70], [180, 180], 'k-')
        ax.plot([70, 290], [180, 400], 'k-')
        ax.plot([70, 70], [0, 56], 'k-')
        ax.plot([70, 400], [56, 320], 'k-')
        ax.plot([180, 180], [0, 70], 'k-')
        ax.plot([180, 400], [70, 70], 'k-')
        ax.plot([240, 240], [70, 180], 'k-')
        ax.plot([240, 400], [180, 180], 'k-')
        ax.plot([130, 180], [0, 70], 'k-')

        # Zone labels
        ax.text(30, 20, 'A', fontsize=10)
        ax.text(30, 150, 'D', fontsize=10)
        ax.text(30, 380, 'E', fontsize=10)
        ax.text(150, 380, 'C', fontsize=10)
        ax.text(160, 20, 'C', fontsize=10)
        ax.text(380, 20, 'E', fontsize=10)
        ax.text(380, 120, 'D', fontsize=10)
        ax.text(380, 260, 'B', fontsize=10)
        ax.text(280, 380, 'B', fontsize=10)

        if print_figure and ax is not None:
            fig = ax.get_figure()
            fig.savefig('Clarke_EGA.png', dpi=300, bbox_inches='tight')

    percentage = np.round((total / n) * 100, 2)
    return total, percentage


def clarke_visualization_table(PH, seed=2024, patient=None, model_name=None, single=False):
    """
        Clarke Error Grid Analysis supports both batch 3x3 subplot mode and single-model single-plot mode.
        - Batch mode: clarke_visualization_table(PH=90)
        - Single plot mode: clarke_visualization_table(PH=90, patient="563", model_name="Glucoformer", single=True)
    """

    patients = ["563", "596"]
    models = [
        "Glucoformer", 
        "Crossformer", 
        "PatchTST", 
        "TimeXer",
        "DLinear", 
        "Informer", 
        "Transformer", 
        "LSTM", 
        "GRU"
    ]

    if single:
        if patient is None or model_name is None:
            raise ValueError("单图模式下必须指定 patient 和 model_name")
        y_path = f'../{model_name}/save_{model_name}_prediction_{seed}seed_{PH}min_{patient}patient/true_values.npy'
        yp_path = f'../{model_name}/save_{model_name}_prediction_{seed}seed_{PH}min_{patient}patient/predicted_values.npy'
        if not (os.path.exists(y_path) and os.path.exists(yp_path)):
            print(f"缺少数据: {y_path} 或 {yp_path}")
            return None
        y = np.load(y_path)[:, -1, 0]
        yp = np.load(yp_path)[:, -1, 0]
        total, percentage = clarke_colored_zones(y, yp, model_name, PH, patient, ax=None)
        results = [{
            "Model": model_name,
            "Patient": patient,
            "A": percentage[0],
            "B": percentage[1],
            "C": percentage[2],
            "D": percentage[3],
            "E": percentage[4]
        }]
        plt.tight_layout()
        plt.show()
        df = pd.DataFrame(results)
        return df

    all_results = []
    for patient in patients:
        results = []  
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        fig.suptitle(f"Clarke Error Grid Analysis (PH={PH}min, Patient {patient})", fontsize=18, y=0.95)
        for i, model_name in enumerate(models):
            row, col = divmod(i, 3)
            ax = axes[row, col]
            y_path = f'../{model_name}/save_{model_name}_prediction_{seed}seed_{PH}min_{patient}patient/true_values.npy'
            yp_path = f'../{model_name}/save_{model_name}_prediction_{seed}seed_{PH}min_{patient}patient/predicted_values.npy'
            # if not (os.path.exists(y_path) and os.path.exists(yp_path)):
            #     ax.axis('off')
            #     continue
            y = np.load(y_path)[:, -1, 0]
            yp = np.load(yp_path)[:, -1, 0]
            _, percentage = clarke_colored_zones(y, yp, model_name, PH, patient, ax=ax)
            results.append({
                "Model": model_name,
                "Patient": patient,
                "A": percentage[0],
                "B": percentage[1],
                "C": percentage[2],
                "D": percentage[3],
                "E": percentage[4]
            })
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
        plt.close(fig)  
        df = pd.DataFrame(results)
        all_results.extend(results)

    return pd.DataFrame(all_results)
    

def clarke_zone_stats(PH, seeds=[2023, 2024, 2025], patients=["563", "596"], models=None):
    """
    Calculates only the percentage of points in Clarke's A, B, C, D, and E zones, supporting multiple random seeds.
    Outputs a pandas DataFrame with mean ± standard deviation.
    Additionally, it calculates statistics for combined zones (A+B and C+D+E) for each subject.
    """
    if models is None:
        models = [
            "Glucoformer", 
            "Crossformer", 
            "PatchTST", 
            "TimeXer",
            "DLinear", 
            "Informer", 
            "Transformer", 
            "LSTM", 
            "GRU"
        ]

    results = []
    for patient in patients:
        for model_name in models:
            percentages = []
            for seed in seeds:
                y_path = f'../{model_name}/save_{model_name}_prediction_{seed}seed_{PH}min_{patient}patient/true_values.npy'
                yp_path = f'../{model_name}/save_{model_name}_prediction_{seed}seed_{PH}min_{patient}patient/predicted_values.npy'
                if not (os.path.exists(y_path) and os.path.exists(yp_path)):
                    continue
                y = np.load(y_path)[:, -1, 0]
                yp = np.load(yp_path)[:, -1, 0]
                _, percentage = clarke_colored_zones(y, yp, model_name, PH, patient, ax=None, draw=False)
                percentages.append(percentage)
            if len(percentages) == 0:
                continue
            percentages = np.stack(percentages, axis=0)  # shape: [seed, 5]
            mean = np.mean(percentages, axis=0)
            std = np.std(percentages, axis=0)
            # 计算A+B和C+D+E的均值和标准差
            ab_mean = mean[0] + mean[1]
            ab_std = np.sqrt(std[0]**2 + std[1]**2)
            cde_mean = mean[2] + mean[3] + mean[4]
            cde_std = np.sqrt(std[2]**2 + std[3]**2 + std[4]**2)
            results.append({
                "Model": model_name,
                "Patient": patient,
                "A": f"{mean[0]:.2f}±{std[0]:.2f}",
                "B": f"{mean[1]:.2f}±{std[1]:.2f}",
                "C": f"{mean[2]:.2f}±{std[2]:.2f}",
                "D": f"{mean[3]:.2f}±{std[3]:.2f}",
                "E": f"{mean[4]:.2f}±{std[4]:.2f}",
                "A+B": f"{ab_mean:.2f}±{ab_std:.2f}",
                "C+D+E": f"{cde_mean:.2f}±{cde_std:.2f}",
            })
    df = pd.DataFrame(results)
    return df


def plot_attention_heatmap(attention_path, patient, sample_idx, head_idx, dec_seg_idx, feature_names=None):
    import seaborn as sns
    import matplotlib.font_manager as fm
    """
    Load, reshape, and visualize a specific attention heatmap.

    Args:
        config (Namespace): Configuration object with model and path info.
        patient (str or int): Patient ID for file path.
        sample_idx (int): Sample index (from the batch) to visualize.
        head_idx (int): Attention head index to visualize.
        dec_seg_idx (int): Decoder segment index to visualize.
        feature_names (list, optional): Names for y-axis labels. Defaults to ['CHO', 'Insulin', 'BG'].
    """
    if feature_names is None:
        feature_names = ['CHO', 'Insulin', 'BG']
    num_features = len(feature_names)

    # 1. Load attention data

    if not os.path.exists(attention_path):
        print(f"Error: Attention file not found at {attention_path}")
        return

    attn_weights = np.load(attention_path)

    # 2. Reshape dimensions
    # Original: (samples, n_heads, dec_seg_num, enc_seg_num)
    # Target: (batch, n_heads, features, dec_seg_num, enc_seg_num)
    try:
        batch_size = attn_weights.shape[0] // num_features
        reshaped_attn = attn_weights.reshape(batch_size, num_features, *attn_weights.shape[1:])
        reshaped_attn = reshaped_attn.transpose(0, 2, 1, 3, 4)
    except ValueError as e:
        print(f"Error reshaping array: {e}. Total samples might not be divisible by num_features.")
        return

    # 3. Select the specific 2D map for plotting
    try:
        attn_map_2d = reshaped_attn[sample_idx, head_idx, :, dec_seg_idx, :]
    except IndexError as e:
        print(f"Error indexing array: {e}. Check if indices are within bounds.")
        print(f"Shape: {reshaped_attn.shape}, Indices: sample={sample_idx}, head={head_idx}, dec_seg={dec_seg_idx}")
        return

    # 4. Plot the heatmap
    plt.figure(figsize=(12, 1.25))
    ax = sns.heatmap(attn_map_2d, cmap='viridis', yticklabels=feature_names, cbar_kws={'pad': 0.02})

    # Configure x-axis to represent time steps (index * 15)
    num_segments = attn_map_2d.shape[1]
    tick_positions = np.arange(num_segments + 1)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([pos * 15 for pos in tick_positions], rotation=0, fontsize=8)

    ax.set_title(f"Attention Map (Patient: {patient}, Query Time Segment: {dec_seg_idx*15+1}-{dec_seg_idx*15+15} min)")
    ax.set_xlabel("Key Time Step (minutes)")
    ax.set_ylabel("Key Dimension")
    plt.yticks(rotation=60)
    plt.show()