
import random
import numpy as np
import torch


class Config:
    csv_path = ''
    seed = 2021
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    batch_size = 6 
    num_epochs = 200
    lr = 1e-3
    input_format = "signals" # "signals" or "features"
    
    Sheet = "Sheet1"
    output = "Stressvalue" # "Failurepoint" "Stressvalue" "Ultimatetensilestress"
    train_csv_path = "C:\\Users\\user\\Desktop\\PSH_resultss\\data\\" + Sheet + "_train\\" + output + ".csv"
    test_csv_path = "C:\\Users\\user\\Desktop\\PSH_resultss\\data\\" + Sheet + "_test\\" + output + ".csv"

    pth_dir = "C:\\Users\\user\\Desktop\\PSH_resultss\\result\\" + Sheet + "\\best_model_" + output + ".pth"
    result_dir = "C:\\Users\\user\\Desktop\\PSH_resultss\\result\\"+ Sheet + "\\test_result_" + output + ".csv"

    Thickness_scaler_path = "C:\\Users\\user\\Desktop\\PSH_resultss\\data\\" + Sheet + "\\Thickness_scaler.save" 
    Input_scaler_path = "C:\\Users\\user\\Desktop\\PSH_resultss\\data\\"+ Sheet + "\\Input_scaler.save" 
    StressValue_scaler_path = "C:\\Users\\user\\Desktop\\PSH_resultss\\data\\" + Sheet + "\\StressValue_scaler.save" 
    Failurepoint_scaler_path = "C:\\Users\\user\\Desktop\\PSH_resultss\\data\\" + Sheet + "\\Failurepoint_scaler.save" 
    Ultimatetensilestress_scaler_path = "C:\\Users\\user\\Desktop\\PSH_resultss\\data\\" + Sheet + "\\Ultimatetensilestress_scaler.save" 

    save_dir_RMSE = "C:\\Users\\user\\Desktop\\PSH_resultss\\result\\" + Sheet + "\\RMSE_" + output + ".csv" 
    save_dir_Prediction = "C:\\Users\\user\\Desktop\\PSH_resultss\\data\\" + Sheet + "\\Prediction_" + output + ".csv"
    save_dir_GroundTruth = "C:\\Users\\user\\Desktop\\PSH_resultss\\data\\" + Sheet + "\\GroundTruth_" + output + ".csv"

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        
        
if __name__ == '__main__':        
    config = Config()
    seed_everything(config.seed)