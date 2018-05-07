import numpy as np 
import pandas as pd 

dataframe = pd.read_csv("train.csv")
data = dataframe.values 
#print(dataframe)
#print(data.shape)

#print(data[0][-1])

test_dataframe = pd.read_csv("test.csv")
test_data = test_dataframe.values 

posture_list = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]

count = 0
for word in posture_list:
    data[:, -1] = np.where(data[:, -1] == word, count, data[:, -1])
    test_data[:, -1] = np.where(test_data[:, -1] == word, count, test_data[:, -1])
    count += 1

print(data[:, -1])

np.save("x_train", data[:, :-1])
np.save("y_train", data[:, -1])

np.save("x_test", test_data[:, :-1])
np.save("y_test", test_data[:, -1])



