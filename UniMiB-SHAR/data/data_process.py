import numpy as np 
from scipy.io import loadmat

acc_data_content = loadmat("acc_data.mat")
acc_labels_content = loadmat("acc_labels.mat")
acc_names_content = loadmat("acc_names.mat")

#print(acc_data_content)
#print(acc_labels_content)
#print(acc_names_content)

acc_data = acc_data_content["acc_data"]
acc_labels = acc_labels_content["acc_labels"]
acc_names = acc_names_content["acc_names"]

acc_names
print(acc_data.shape)
print(acc_labels.shape)
print(acc_names.shape)

# save as csv to read 
np.savetxt("acc_data.csv", acc_data, delimiter=",")


# 所有data都放在這裡
full_data_content = loadmat("full_data.mat")
full_data = full_data_content["full_data"]

#看形狀很重要
print(full_data.shape)

print(full_data)
