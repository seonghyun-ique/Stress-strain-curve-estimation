
#%%
import torch 
import numpy as np 
from sklearn.metrics import mean_squared_error

class Meter:
    def __init__(self, n_classes=5):
        self.metrics = {}
        # self.predictions = torch.tensor([], dtype=torch.float) # 예측값을 저장하는 텐서.
        # self.actual = torch.tensor([], dtype=torch.float) # 실제값을 저장하는 텐서.
    
    def update(self, x, y, loss):
        # x = x.detach().cpu() # 넘파이 배열로 변경.
        # y = y.detach().cpu() # 넘파이 배열로 변경.
        # self.predictions = torch.cat((self.predictions, x), 0) # cat함수를 통해 예측값을 누적.
        # self.actual = torch.cat((self.actual, y), 0) # cat함수를 통해 실제값을 누적.
        # predictions = self.predictions.numpy() # 넘파이 배열로 변경.
        # actual = self.actual.numpy() # 넘파이 배열로 변경.

        predictions = x.detach().cpu().numpy() # 넘파이 배열로 변경.
        actual = y.detach().cpu().numpy() # 넘파이 배열로 변경.
        rmse = np.sqrt(mean_squared_error(predictions, actual)) # sklearn을 이용해 RMSE를 계산.
        # print("predictions: ",  predictions[0])
        # print("actual: ",  actual[0])
        self.metrics['rmse'] += rmse
        
        loss = loss # 넘파이 배열로 변경.

        self.metrics['loss'] += loss
        
       
    
    def init_metrics(self):
        self.metrics['loss'] = 0
        self.metrics['rmse'] = 0

        
    def get_metrics(self):
        return self.metrics

   
if __name__ == '__main__':
    meter = Meter()  
# %%
# x= torch.tensor([1,1,1,1])
# y = torch.tensor([2,2,2,2])
# loss = 0.2
# meters = Meter()
# meters.init_metrics()
# meters.update(x,y,loss)

# # %%
# meters.get_metrics()

# # %%
# x= torch.tensor([1,1,1,1])
# y = torch.tensor([2,2,2,2])
# loss = 0.2
# meters.update(x,y,loss)
# meters.get_metrics()
# # %%
