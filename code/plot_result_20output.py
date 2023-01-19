import os 
import numpy as np 
import matplotlib.pyplot as plt

sheet = "Sheet4"
output = "Failurepoint" # Failurepoint, Ultimatetensilestress, Stressvalue

groundtruth_dir = "C:\\Users\\CJY\\stressprediction\\result\\" + sheet + "\\GroundTruth_" + output + ".csv"
prediction_dir = "C:\\Users\\CJY\\stressprediction\\result\\" + sheet + "\\Prediction_" + output + ".csv"
rmse_dir = "C:\\Users\\CJY\\stressprediction\\result\\" + sheet + "\\RMSE_" + output + ".csv"

save_Dir = "C:\\Users\\CJY\\stressprediction\\result\\" + sheet + "\\RMSEboxplot_" + output + ".png"

groundtruth_data = np.loadtxt(groundtruth_dir, delimiter=",",skiprows=1, usecols=range(1,3))
prediction_data = np.loadtxt(prediction_dir, delimiter=",",skiprows=1, usecols=range(1,3))
rmse_data = np.loadtxt(rmse_dir, delimiter=",",skiprows=1, usecols=range(1,3))


#normalized_rmse = rmse_data / groundtruth_data * 100
normalized_rmse = rmse_data 

# 1. 기본 스타일 설정
plt.style.use('default')
plt.rcParams['figure.figsize'] = (4, 3)
plt.rcParams['font.size'] = 12

# 2. 데이터 준비
np.random.seed(0)
data_a = np.random.normal(0, 2.0, 1000)
data_b = np.random.normal(-3.0, 1.5, 500)
data_c = np.random.normal(1.2, 1.5, 1500)

# 3. 그래프 그리기
fig, ax = plt.subplots()

ax.boxplot([normalized_rmse[:,0], normalized_rmse[:,1]])
ax.set_ylim(-10.0, 150.0)
ax.set_xlabel('Data')
ax.set_ylabel('RMSE')

print(save_Dir)
plt.show()
