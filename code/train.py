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


class Trainer:
    def __init__(self, net, lr, batch_size, num_epochs, input_format, outputdata):
        self.net = net.to(config.device)
        self.num_epochs = num_epochs
        self.criterion = nn.MSELoss()
        self.optimizer = AdamW(self.net.parameters(), lr=lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=5e-6)
        self.best_loss = float('inf')
        self.phases = ['train', 'val']
        self.dataloaders = {
            phase: get_dataloader(phase, input_format, outputdata, batch_size) for phase in self.phases
        }
        self.train_df_logs = pd.DataFrame()
        self.val_df_logs = pd.DataFrame()

            
    def _train_epoch(self, phase):
        print(f"{phase} mode | time: {time.strftime('%H:%M:%S')}")
        
        self.net.train() if phase == 'train' else self.net.eval()
        meter = Meter()
        meter.init_metrics()
        
        for i, (data, target) in enumerate(self.dataloaders[phase]):
            data = data.to(config.device)
            target = target.to(config.device)
            
            output = self.net(data)
            
            loss = self.criterion(output, target)
            #print("loss: ", loss.item())
                        
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # print("data shape: ", data.shape)
            # print("output shape: ", output.shape)
            # print("target shape: ", target.shape)
            # print("output: ", output)
            # print("target :", target)

            meter.update(output, target, loss.item())
        
        metrics = meter.get_metrics()
        metrics = {k:v / i for k, v in metrics.items()}
        df_logs = pd.DataFrame([metrics])
        
        if phase == 'train':
            self.train_df_logs = pd.concat([self.train_df_logs, df_logs], axis=0)
        else:
            self.val_df_logs = pd.concat([self.val_df_logs, df_logs], axis=0)
        
        # show logs
        print('{}: {}, {}: {}'
              .format(*(x for kv in metrics.items() for x in kv))
             )

        
        return loss
    
    def run(self, pth_dir):
        for epoch in range(self.num_epochs):
            self._train_epoch(phase='train')
            with torch.no_grad():
                val_loss = self._train_epoch(phase='val')
                #self.scheduler.step()
            
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                print('\nNew checkpoint\n')
                self.best_loss = val_loss
                torch.save(self.net.state_dict(), pth_dir)

              
if __name__ == '__main__':
    # init config and set random seed
    config = Config()
    seed_everything(config.seed)
     
    # init model
    model = RNNAttentionModel(1, 64, 'lstm', False)
    #model = RNNModel(1, 64, 'lstm', True)
    #model = CNN(num_classes=30, hid_size=128)  
    #model = Regressor()         

    # start train
    trainer = Trainer(net=model, lr=config.lr, batch_size=config.batch_size, num_epochs=config.num_epochs, input_format = config.input_format, outputdata = config.output)
    trainer.run(pth_dir = config.pth_dir)  
              
    # write logs
    train_logs = trainer.train_df_logs
    train_logs.columns = ["train_"+ colname for colname in train_logs.columns]
    val_logs = trainer.val_df_logs
    val_logs.columns = ["val_"+ colname for colname in val_logs.columns]

    logs = pd.concat([train_logs,val_logs], axis=1)
    logs.reset_index(drop=True, inplace=True)
    logs = logs.loc[:, [
        'train_loss', 'val_loss', 
        'train_rmse', 'val_rmse']
                                     ]
    print(logs.head())
    logs.to_csv('cnn.csv', index=False)