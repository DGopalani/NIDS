import csv
import pandas as pd
import os 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import time 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler 

#UDP training dataset
UDP_csv_file_path = 'Data/01-12/DrDoS_UDP.csv'
UDP_df = pd.read_csv(UDP_csv_file_path)
UDP_df[' Label'] = UDP_df[' Label'].map({'BENIGN': 0, 'UDP-lag': 1, 'DrDoS_UDP': 2, 'Syn': 3, 'WebDDoS': 4, 'DrDoS_MSSQL': 5})
UDP_vars_to_keep = [' Destination Port', ' Fwd Packet Length Std', ' Packet Length Std', ' min_seg_size_forward', ' Protocol', ' Label']
UDP_df_filtered = UDP_df[UDP_vars_to_keep]

#UDP test dataset
UDP_test_csv_file_path = 'Data/03-11/UDP.csv'
UDP_test_df = pd.read_csv(UDP_test_csv_file_path)
#apply feature mappin on label column
UDP_test_df[' Label'] = UDP_test_df[' Label'].map({'BENIGN': 0, 'UDPLag': 1, 'UDP': 2, 'Syn': 3, 'WebDDoS': 4, 'MSSQL': 5})
#make testing filtered df based on vars_to_keep
UDP_test_df_filtered = UDP_test_df[UDP_vars_to_keep]
#keep only 0, 1, and 4 in Label column of test_df_filtered
UDP_test_df_filtered = UDP_test_df_filtered[UDP_test_df_filtered[' Label'].isin([0, 2, 4])]



#Train on Random Forest Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

print('')
print("UDP")
#Prepare UDP train data
UDP_y_train = UDP_df_filtered[[' Label']]
UDP_X_train = UDP_df_filtered.drop(columns=[' Label'], axis=1)
UDP_sc = MinMaxScaler() 
UDP_X_train = UDP_sc.fit_transform(UDP_X_train)

#Prepare UDP test data
UDP_y_test = UDP_test_df_filtered[[' Label']]
UDP_X_test = UDP_test_df_filtered.drop(columns=[' Label'], axis=1)
UDP_X_test = UDP_sc.fit_transform(UDP_X_test)

#print data shapes
print(UDP_X_train.shape, UDP_X_test.shape) 
print(UDP_y_train.shape, UDP_y_test.shape) 

def run_model():
    #Train on UDP data
    clfr = RandomForestClassifier(n_estimators = 5)
    start_time = time.time()
    clfr.fit(UDP_X_train, UDP_y_train.values.ravel())
    end_time = time.time()
    print("Training time: ", end_time-start_time)

    #Test on Random Forest Model with UDP data
    start_time = time.time()
    UDP_y_test_pred = clfr.predict(UDP_X_test)
    end_time = time.time()
    print("Testing time: ", end_time-start_time)

    #Print UDP results
    print("Train score is:", clfr.score(UDP_X_train, UDP_y_train))
    print("Test score is:", clfr.score(UDP_X_test, UDP_y_test))
    #micro=Calculate metrics globally by counting the total true positives, false negatives and false positives.
    precision = precision_score(UDP_y_test, UDP_y_test_pred, average='macro')
    recall = recall_score(UDP_y_test, UDP_y_test_pred, average='macro')
    conf_matrix = confusion_matrix(UDP_y_test, UDP_y_test_pred)
    false_positive_rate = conf_matrix[0, 1] / (conf_matrix[0, 0] + conf_matrix[0, 1])
    print("Precision:", precision)
    print("Recall:", recall)
    print("False Positive Rate:", false_positive_rate)
    print("Confusion Matrix:")
    print(conf_matrix)

