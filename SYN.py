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

#SYN training dataset
SYN_csv_file_path = 'Data/01-12/Syn.csv'
SYN_df = pd.read_csv(SYN_csv_file_path)
SYN_df[' Label'] = SYN_df[' Label'].map({'BENIGN': 0, 'UDP-lag': 1, 'DrDoS_UDP': 2, 'Syn': 3, 'WebDDoS': 4, 'DrDoS_MSSQL': 5})
SYN_vars_to_keep = [' ACK Flag Count', 'Init_Win_bytes_forward', ' min_seg_size_forward', 'Fwd IAT Total', ' Flow Duration', ' Label']
SYN_df_filtered = SYN_df[SYN_vars_to_keep]

#SYN test dataset
SYN_test_csv_file_path = 'Data/03-11/Syn.csv'
SYN_test_df = pd.read_csv(SYN_test_csv_file_path)

#apply feature mappin on label column
SYN_test_df[' Label'] = SYN_test_df[' Label'].map({'BENIGN': 0, 'UDPLag': 1, 'UDP': 2, 'Syn': 3, 'WebDDoS': 4, 'MSSQL': 5, 'LDAP': 6})
#make testing filtered df based on vars_to_keep
SYN_test_df_filtered = SYN_test_df[SYN_vars_to_keep]
#keep only 0 and 3 in Label column of test_df_filtered
SYN_test_df_filtered = SYN_test_df_filtered[SYN_test_df_filtered[' Label'].isin([0, 3])]

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
print("SYN")
#Prepare SYN train data
SYN_y_train = SYN_df_filtered[[' Label']]
SYN_X_train = SYN_df_filtered.drop(columns=[' Label'], axis=1)
SYN_sc = MinMaxScaler()
SYN_X_train = SYN_sc.fit_transform(SYN_X_train)

#Prepare SYN test data
SYN_y_test = SYN_test_df_filtered[[' Label']]
SYN_X_test = SYN_test_df_filtered.drop(columns=[' Label'], axis=1)
SYN_sc = MinMaxScaler()
SYN_X_test = SYN_sc.fit_transform(SYN_X_test)

#print data shapes
print(SYN_X_train.shape, SYN_X_test.shape)
print(SYN_y_train.shape, SYN_y_test.shape)

def run_model():
    #Train on SYN data
    clfr = RandomForestClassifier(n_estimators = 5)
    start_time = time.time()
    clfr.fit(SYN_X_train, SYN_y_train.values.ravel())
    end_time = time.time()
    print("Training time: ", end_time-start_time)

    #Test on Random Forest Model with SYN data
    start_time = time.time()
    SYN_y_test_pred = clfr.predict(SYN_X_test)
    end_time = time.time()
    print("Testing time: ", end_time-start_time)

    #Print SYN results
    print("Train score is:", clfr.score(SYN_X_train, SYN_y_train))
    print("Test score is:", clfr.score(SYN_X_test, SYN_y_test))
    #micro=Calculate metrics globally by counting the total true positives, false negatives and false positives.
    precision = precision_score(SYN_y_test, SYN_y_test_pred, average='macro')
    recall = recall_score(SYN_y_test, SYN_y_test_pred, average='macro')
    conf_matrix = confusion_matrix(SYN_y_test, SYN_y_test_pred)
    false_positive_rate = conf_matrix[0, 1] / (conf_matrix[0, 0] + conf_matrix[0, 1])
    print("Precision:", precision)
    print("Recall:", recall)
    print("False Positive Rate:", false_positive_rate)
    print("Confusion Matrix:")
    print(conf_matrix)


