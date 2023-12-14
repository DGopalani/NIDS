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

#LDAP training dataset
LDAP_csv_file_path = 'Data/01-12/DrDoS_LDAP.csv'
LDAP_df = pd.read_csv(LDAP_csv_file_path)
LDAP_df[' Label'] = LDAP_df[' Label'].map({'BENIGN': 0, 'UDP-lag': 1, 'UDP': 2, 'Syn': 3, 'WebDDoS': 4, 'DrDoS_MSSQL': 5, 'DrDoS_LDAP': 6})
LDAP_vars_to_keep = [' Max Packet Length', ' Fwd Packet Length Max', ' Fwd Packet Length Min', ' Average Packet Size', ' Min Packet Length', ' Label']
LDAP_df_filtered = LDAP_df[LDAP_vars_to_keep]

#LDAP test dataset
LDAP_test_csv_file_path = 'Data/03-11/LDAP.csv'
LDAP_test_df = pd.read_csv(LDAP_test_csv_file_path)
#apply feature mapping on label column
LDAP_test_df[' Label'] = LDAP_test_df[' Label'].map({'BENIGN': 0, 'UDPLag': 1, 'UDP': 2, 'Syn': 3, 'WebDDoS': 4, 'MSSQL': 5, 'LDAP': 6, 'NetBIOS': 7})
#make testing filtered df based on vars_to_keep
LDAP_test_df_filtered = LDAP_test_df[LDAP_vars_to_keep]
#keep only 0, 6 in Label column of test_df_filtered
LDAP_test_df_filtered = LDAP_test_df_filtered[LDAP_test_df_filtered[' Label'].isin([0, 6])]

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
print("LDAP")
#Prepare LDAP train data
LDAP_y_train = LDAP_df_filtered[[' Label']]
LDAP_X_train = LDAP_df_filtered.drop(columns=[' Label'], axis=1)
LDAP_sc = MinMaxScaler()
LDAP_X_train = LDAP_sc.fit_transform(LDAP_X_train)

#Prepare LDAP test data
LDAP_y_test = LDAP_test_df_filtered[[' Label']]
LDAP_X_test = LDAP_test_df_filtered.drop(columns=[' Label'], axis=1)
LDAP_sc = MinMaxScaler()
LDAP_X_test = LDAP_sc.fit_transform(LDAP_X_test)

#print data shapes
print(LDAP_X_train.shape, LDAP_X_test.shape)
print(LDAP_y_train.shape, LDAP_y_test.shape)

def run_model():
    #Train on LDAP data
    clfr = RandomForestClassifier(n_estimators = 5)
    start_time = time.time()
    clfr.fit(LDAP_X_train, LDAP_y_train.values.ravel())
    end_time = time.time()
    print("Training time: ", end_time-start_time)

    #Test on Random Forest Model with LDAP data
    start_time = time.time()
    LDAP_y_test_pred = clfr.predict(LDAP_X_test)
    end_time = time.time()
    print("Testing time: ", end_time-start_time)

    #Print LDAP results
    print("Train score is:", clfr.score(LDAP_X_train, LDAP_y_train))
    print("Test score is:", clfr.score(LDAP_X_test, LDAP_y_test))
    #micro=Calculate metrics globally by counting the total true positives, false negatives and false positives.
    precision = precision_score(LDAP_y_test, LDAP_y_test_pred, average='macro')
    recall = recall_score(LDAP_y_test, LDAP_y_test_pred, average='macro')
    conf_matrix = confusion_matrix(LDAP_y_test, LDAP_y_test_pred)
    false_positive_rate = conf_matrix[0, 1] / (conf_matrix[0, 0] + conf_matrix[0, 1])
    print("Precision:", precision)
    print("Recall:", recall)
    print("False Positive Rate:", false_positive_rate)
    print("Confusion Matrix:")
    print(conf_matrix)
