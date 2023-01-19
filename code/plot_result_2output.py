#%%
import os 
import numpy as np 
import matplotlib.pyplot as plt

output = "Failurepoint" # Failurepoint, Ultimatetensilestress

groundtruth_sheet1_dir = "C:\\Users\\CJY\\stressprediction\\result\\Sheet1" + "\\groundtruth_" + output + ".csv"
groundtruth_sheet2_dir = "C:\\Users\\CJY\\stressprediction\\result\\Sheet2" + "\\groundtruth_" + output + ".csv"
groundtruth_sheet3_dir = "C:\\Users\\CJY\\stressprediction\\result\\Sheet3" + "\\groundtruth_" + output + ".csv"
groundtruth_sheet4_dir = "C:\\Users\\CJY\\stressprediction\\result\\Sheet4" + "\\groundtruth_" + output + ".csv"

rmse_sheet1_dir = "C:\\Users\\CJY\\stressprediction\\result\\Sheet1" + "\\RMSE_" + output + ".csv"
rmse_sheet2_dir = "C:\\Users\\CJY\\stressprediction\\result\\Sheet2" + "\\RMSE_" + output + ".csv"
rmse_sheet3_dir = "C:\\Users\\CJY\\stressprediction\\result\\Sheet3" + "\\RMSE_" + output + ".csv"
rmse_sheet4_dir = "C:\\Users\\CJY\\stressprediction\\result\\Sheet4" + "\\RMSE_" + output + ".csv"

save_Dir = "C:\\Users\\CJY\\stressprediction\\result\\" + "RMSE_" + output + ".png"

groundtruth_sheet1_data = np.loadtxt(groundtruth_sheet1_dir, delimiter=",",skiprows=1, usecols=range(1,3))
groundtruth_sheet2_data = np.loadtxt(groundtruth_sheet2_dir, delimiter=",",skiprows=1, usecols=range(1,3))
groundtruth_sheet3_data = np.loadtxt(groundtruth_sheet3_dir, delimiter=",",skiprows=1, usecols=range(1,3))
groundtruth_sheet4_data = np.loadtxt(groundtruth_sheet4_dir, delimiter=",",skiprows=1, usecols=range(1,3))

rmse_sheet1_data = np.loadtxt(rmse_sheet1_dir, delimiter=",",skiprows=1, usecols=range(1,3))
rmse_sheet2_data = np.loadtxt(rmse_sheet2_dir, delimiter=",",skiprows=1, usecols=range(1,3))
rmse_sheet3_data = np.loadtxt(rmse_sheet3_dir, delimiter=",",skiprows=1, usecols=range(1,3))
rmse_sheet4_data = np.loadtxt(rmse_sheet4_dir, delimiter=",",skiprows=1, usecols=range(1,3))


#normalized_rmse_sheet1 = rmse_sheet1_data / groundtruth_sheet1_data * 100
#normalized_rmse_sheet2 = rmse_sheet2_data / groundtruth_sheet2_data * 100
#normalized_rmse_sheet3 = rmse_sheet3_data / groundtruth_sheet3_data * 100
#normalized_rmse_sheet4 = rmse_sheet4_data / groundtruth_sheet4_data * 100
normalized_rmse_sheet1 = rmse_sheet1_data 
normalized_rmse_sheet2 = rmse_sheet2_data 
normalized_rmse_sheet3 = rmse_sheet3_data 
normalized_rmse_sheet4 = rmse_sheet4_data 

mean_rmse_sheet1 = np.average(normalized_rmse_sheet1, axis=0)
mean_rmse_sheet2 = np.average(normalized_rmse_sheet2, axis=0)
mean_rmse_sheet3 = np.average(normalized_rmse_sheet3, axis=0)
mean_rmse_sheet4 = np.average(normalized_rmse_sheet4, axis=0)

# Plot strain
x = np.arange(4)
sheets = ["Sheet1", "Sheet2", "Sheet3", "Sheet4"]
values = [mean_rmse_sheet1[0], mean_rmse_sheet2[0], mean_rmse_sheet3[0], mean_rmse_sheet4[0]]

plt.bar(x, values)
plt.xticks(x, sheets)

plt.show()



# Plot stress
x = np.arange(4)
sheets = ["Sheet1", "Sheet2", "Sheet3", "Sheet4"]
values = [mean_rmse_sheet1[1], mean_rmse_sheet2[1], mean_rmse_sheet3[1], mean_rmse_sheet4[1]]

plt.bar(x, values)
plt.xticks(x, sheets)

plt.show()


