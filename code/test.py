import os
import time
import random

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import (CosineAnnealingLR,
                                      CosineAnnealingWarmRestarts,
                                      StepLR,
                                      ExponentialLR)

from meter import Meter
from dataset import get_dataloader
from models import *
from config import Config, seed_everything
from sklearn.externals import joblib

class Evaluater:
    def __init__(self, net, batch_size, pth_dir, save_dir, input_format, outputdata):
        self.net = net.to(config.device)
        self.pth_dir = pth_dir
        self.save_dir = save_dir
        self.phases = ['test']
        self.dataloaders = {
            phase: get_dataloader(phase, input_format, outputdata, batch_size) for phase in self.phases
        }
        self.RMSE = pd.DataFrame()
        self.Prediction = pd.DataFrame()
        self.GroundTruth = pd.DataFrame()

        if outputdata == 'Stressvalue': 
            self.scaler = joblib.load(Config.StressValue_scaler_path) 
        elif outputdata == 'Failurepoint':
            self.scaler = joblib.load(Config.Failurepoint_scaler_path) 
        elif outputdata == 'Ultimatetensilestress':
            self.scaler = joblib.load(Config.Ultimatetensilestress_scaler_path) 

    
    def _test_epoch(self, phase):
        print(f"{phase} mode | time: {time.strftime('%H:%M:%S')}")
        self.net.load_state_dict(torch.load(self.pth_dir))
        self.net.eval() 
        
        for i, (data, target) in enumerate(self.dataloaders[phase]):
            data = data.to(config.device)
            target = target.to(config.device)
            
            output = self.net(data)
            output = self.scaler.inverse_transform(output.cpu().detach().numpy())   
            target = self.scaler.inverse_transform(target.cpu().detach().numpy())                  
            diff = target - output
            
            self.RMSE = pd.concat([self.RMSE, pd.DataFrame(np.sqrt(diff**2))])
            self.Prediction = pd.concat([self.Prediction, pd.DataFrame(output)])
            self.GroundTruth = pd.concat([self.GroundTruth, pd.DataFrame(target)])

            # print("data shape: ", data.shape)
            # print("output shape: ", output.shape)
            # print("target shape: ", target.shape)

        
        return self.RMSE, self.Prediction, self.GroundTruth
    
    def run(self):
        RMSE, Prediction, GroundTruth = self._test_epoch(phase='test')
        print("RMSE : ", RMSE)
        save_dir_RMSE = Config.save_dir_RMSE
        save_dir_Prediction = Config.save_dir_Prediction
        save_dir_GroundTruth = Config.save_dir_GroundTruth
        RMSE.to_csv(save_dir_RMSE)
        Prediction.to_csv(save_dir_Prediction)
        GroundTruth.to_csv(save_dir_GroundTruth)

              
if __name__ == '__main__':
    # init config and set random seed
    config = Config()
    seed_everything(config.seed)
     
    # init model
    model = RNNAttentionModel(1, 64, 'lstm', False)
    #model = RNNModel(1, 64, 'lstm', True)
    #model = CNN(num_classes=1, hid_size=128)  
    #model = Regressor()         

    # start train
    evaluaner = Evaluater(net=model, batch_size=config.batch_size, pth_dir = config.pth_dir, save_dir = config.result_dir, input_format = config.input_format, outputdata = config.output)
    evaluaner.run()  
              
    # write logs
