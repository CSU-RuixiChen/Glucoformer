import os, warnings
MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
import pandas as pd
import numpy as np
import torch
from datetime import datetime
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter1d

try:
    from Glucose_Data.timefeatures import time_features
except ImportError:
    from timefeatures import time_features

warnings.filterwarnings(action='ignore')


####################################
#            ShanghaiDM            #
####################################
class IOB:
    def __init__(self):
        self.x1, self.x2 = 0, 0  
        self.dx1, self.dx2 = 0, 0 
        self.kdia = 0.025 

    def step(self, insulin):
        dx1 = insulin - self.kdia * self.x1
        dx2 = self.kdia * self.x1 - self.kdia * self.x2
        self.x1, self.x2 = self.x1 + dx1, self.x2 + dx2 # euler

    def get_IOB(self):
        return self.x1+self.x2


def get_iob(data):
    IOB_calculator = IOB()
    ins = data.INS # 5분단위의 SMB
    ins_1m, iob_1m = np.zeros([len(ins)*5]), np.zeros([len(ins)*5])
    
    ins_1m = [ins[i//5] if i%5==0 else 0 for i, _ in enumerate(ins_1m)] # 1분단위로 변경
    for i, _ins in enumerate(ins_1m): # IOB 계산
        IOB_calculator.step(_ins)
        iob = IOB_calculator.get_IOB()
        iob_1m[i] = iob_1m[i] + iob

    iob_5m = iob_1m[::5]
    return iob_5m

# Function to remove each day which has more than 20 min consecutive CGM missings
def remove_consecutive_nans(df, column, n, date_column):
    df = df.reset_index()
    df['group'] = (df[column].notna().cumsum())
    df['nan_count'] = df.groupby('group')[column].transform(lambda x: x.isna().sum())
    
    days_missing = df[(df.nan_count>=n+1) & (pd.isna(df[column]))][date_column].unique()
    df_cleaned = df[~df[date_column].isin(days_missing)]
    
    return df_cleaned

# Function to filter the DataFrame by valid dates
def filter_by_date(df, valid_dates, date_column):
    return df[df[date_column].dt.floor('d').isin(valid_dates)]


def preprocess_Shanghai(main_dir,  Use_IOB=False):
    # Loop through each dataset type (T1DM and T2DM)
    dataset_types = ['Shanghai_T1DM', 'Shanghai_T2DM']

    for dataset_type in dataset_types:
        fp = os.path.join(main_dir, 'Shanghai_Dataset', dataset_type, 'Raw') + os.sep
        new_fp = fp.replace('Raw', 'Processed')
        
        if not os.path.exists(new_fp):
            os.makedirs(new_fp)
            
        file_list = os.listdir(fp)
        for i, filename in enumerate(file_list):
            if filename.endswith('.csv'):
                data_new = pd.read_csv(fp+filename)
                subject = filename.split('_')[0]+ '_' +filename.split('_')[1]
                print(f"[{i+1}] Original File Name: {filename} ")
            
                data_new['Date'] = pd.to_datetime(data_new['Date'])
                data_new.set_index('Date', inplace=True)
            
                # Resampling to 5-minute intervals
                resampled_data_5min = data_new.resample('5min').first()
                resampled_data_5min['is_interpolated'] = [True if pd.isna(cgm) else False for cgm in resampled_data_5min['CGM (mg / dl)']] 
                
                resampled_data_5min = resampled_data_5min.rename(columns={'CSII - basal insulin (Novolin R, IU / H)': 'Basal'})
                resampled_data_5min = resampled_data_5min.rename(columns={'CSII - bolus insulin (Novolin R, IU)': 'Bolus'})
                resampled_data_5min = resampled_data_5min.rename(columns={'CGM (mg / dl)': 'CGM'})
            
                # Interpolating CGM values
                resampled_data_5min['CGM'] = resampled_data_5min['CGM'].interpolate(method = 'cubic')
                resampled_data_5min.loc[resampled_data_5min['CGM']<40, 'CGM'] = 40
                resampled_data_5min.loc[resampled_data_5min['CGM']>400, 'CGM'] = 400
                
                # Forward filling Basal values as before
                resampled_data_5min['Basal'] = resampled_data_5min['Basal'].fillna(method='ffill')
                
                # Rounding Bolus values to the nearest 5 minutes and filling NaNs with 0
                resampled_data_5min['Bolus'] = resampled_data_5min['Bolus'].fillna(0)
                
                # Convert 'Basal' from units/hour to units/5 minutes
                resampled_data_5min['Basal per 5min'] = resampled_data_5min['Basal'] / 12
                
                # Creating a new column 'INS' that adds 'Bolus' and 'Basal per 5min' values
                resampled_data_5min['INS'] = resampled_data_5min['Bolus'] + resampled_data_5min['Basal per 5min']
                
                # Resetting the index to make 'Date' a column again
                resampled_data_5min.reset_index(inplace=True)
                
                # Define the file path for saving the CSV file
                # New CSV filename using the extracted number
                new_filename = f'{new_fp}5minResamepled_{subject}.csv'
                print(f"--> [Saved] New File Name: {new_filename} ")
                resampled_data_5min.to_csv(new_filename, index=False)

        if  Use_IOB:
            print(f"[IOB Transformation & Combine to one .pt file and Save]", end = ' ')
            file_list = os.listdir(new_fp)                
            all_data = []
            for i, file in enumerate(file_list):
                if (os.path.splitext(file)[1]=='.csv'):        
                    sid = file.split('.')[0].split('_')[1]+'_'+file.split('.')[0].split('_')[2]
                    
                    _data = pd.read_csv(new_fp+file)
                    data = _data.loc[:, ['Date', 'CGM', 'INS', 'Bolus', 'Basal per 5min', 'is_interpolated']].rename(columns = {'Date': 'date'})
                    data[