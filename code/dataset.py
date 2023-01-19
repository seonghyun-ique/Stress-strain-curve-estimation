#%%
import numpy as np 

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

from config import Config
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

class SignalData(Dataset):

    def __init__(self, x_data, y_data):
        self.x_data = torch.FloatTensor(x_data).unsqueeze(0)
        self.y_data = torch.FloatTensor(y_data)
        self.len = self.y_data.shape[0]

    def __getitem__(self, index):

        return self.x_data[:, index], self.y_data[index] 

    def __len__(self):
        return self.len

class FeatureData(Dataset):

    def __init__(self, x_data, y_data):
        self.x_data = torch.FloatTensor(x_data)
        self.y_data = torch.FloatTensor(y_data)
        self.len = self.y_data.shape[0]

    def __getitem__(self, index):

        return self.x_data[index], self.y_data[index] 

    def __len__(self):
        return self.len
      
def get_dataloader(phase, input_format, output, batch_size = 96) -> DataLoader:
    '''
    Dataset and DataLoader.
    Parameters:
        pahse: training or validation phase.
        batch_size: data per iteration.
    Returns:
        data generator
    '''
    if phase == 'train' : 
        df = pd.read_csv(Config.train_csv_path)
    elif phase =="val" : 
        df = pd.read_csv(Config.train_csv_path)
    elif phase == 'test': 
        df = pd.read_csv(Config.test_csv_path)

    if output == 'Stressvalue': 
        # Thickness = np.repeat(df.iloc[:,0].values.reshape([-1,1]), 1000, axis=1)
        # Thickness_scaler = joblib.load(Config.Thickness_scaler_path) 
        # Thickness = Thickness_scaler.transform(Thickness)
        X = df.iloc[:, 1:-20].values
        Input_scaler = joblib.load(Config.Input_scaler_path) 
        X = Input_scaler.transform(X)
        
        Y = df.iloc[:,-20:].values  
        StressValue_scaler = joblib.load(Config.StressValue_scaler_path) 
        Y = StressValue_scaler.transform(Y)

    elif output == 'Failurepoint':
        X = df.iloc[:, 1:-2].values
        Input_scaler = joblib.load(Config.Input_scaler_path) 
        X = Input_scaler.transform(X)

        Y = df.iloc[:,-2:].values  
        StressValue_scaler = joblib.load(Config.Failurepoint_scaler_path) 
        Y = StressValue_scaler.transform(Y)

    elif output == 'Ultimatetensilestress':
        X = df.iloc[:, 1:-2].values
        Input_scaler = joblib.load(Config.Input_scaler_path) 
        X = Input_scaler.transform(X)

        Y = df.iloc[:,-2:].values  
        StressValue_scaler = joblib.load(Config.Ultimatetensilestress_scaler_path) 
        Y = StressValue_scaler.transform(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, shuffle = True, random_state=Config.seed)

    if input_format == "signals": 
        if phase == 'train': 
            dataset = SignalData(X_train, Y_train)
        elif phase =="val" : 
            dataset = SignalData(X_test, Y_test)
        elif phase =="test" : 
            dataset = SignalData(X, Y)

    elif input_format == "features": 
        if phase == 'train': 
            dataset = FeatureData(X_train, Y_train)
        elif phase =="val" : 
            dataset = FeatureData(X_test, Y_test)
        elif phase =="test" : 
            dataset = FeatureData(X, Y)

    #return dataset
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=0)
    return dataloader
  
  
if __name__ == '__main__':
    #train_dataloader = get_dataloader(phase='train', input_format = "signals",  batch_size=2)
    #val_dataloader = get_dataloader(phase='val', input_format = "signals", batch_size=2)
    test_dataloader = get_dataloader(phase='train', input_format = "signals", output ="Stressvalue", batch_size=2)

# # %%
# batch_iterator = iter(test_dataloader)
# x, y = next(batch_iterator)

# # %%
# print(x)
# print(x.shape)
# # %%
# print(y)
# print(y.shape)
# # %%

# # %%
