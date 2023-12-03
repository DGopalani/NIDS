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

#MSSQL training dataset
MSSQL_csv_file_path = 'Data/01-12/DrDoS_MSSQL.csv'
MSSQL_df = pd.read_csv(MSSQL_csv_file_path)
MSSQL_df[' Label'] = MSSQL_df[' Label'].map({'BENIGN': 0, 'UDP-lag': 1, 'DrDoS_UDP': 2, 'Syn': 3, 'WebDDoS': 4, 'DrDoS_MSSQL': 5})
MSSQL_vars_to_keep = ['Fwd Packets/s', ' Protocol', ' Label']
MSSQL_df_filtered = MSSQL_df[MSSQL_vars_to_keep]

#MSSQL test dataset
MSSQL_test_csv_file_path = 'Data/03-11/MSSQL.csv'
MSSQL_test_df = pd.read_csv(MSSQL_test_csv_file_path)
#apply feature mappin on label column
MSSQL_test_df[' Label'] = MSSQL_test_df[' Label'].map({'BENIGN': 0, 'UDPLag': 1, 'UDP': 2, 'Syn': 3, 'WebDDoS': 4, 'MSSQL': 5, 'LDAP': 6})
#make testing filtered df based on vars_to_keep
MSSQL_test_df_filtered = MSSQL_test_df[MSSQL_vars_to_keep]
#keep only 0, 1, and 4 in Label column of test_df_filtered
MSSQL_test_df_filtered = MSSQL_test_df_filtered[MSSQL_test_df_filtered[' Label'].isin([0, 1, 2, 4, 5])]

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
print("MSSQL")
#Prepare MSSQL train data
MSSQL_y_train = MSSQL_df_filtered[[' Label']]
MSSQL_X_train = MSSQL_df_filtered.drop(columns=[' Label'], axis=1)
MSSQL_sc = MinMaxScaler()
MSSQL_X_train = MSSQL_sc.fit_transform(MSSQL_X_train)

#Prepare MSSQL test data
MSSQL_y_test = MSSQL_test_df_filtered[[' Label']]
MSSQL_X_test = MSSQL_test_df_filtered.drop(columns=[' Label'], axis=1)
MSSQL_sc = MinMaxScaler()
MSSQL_X_test = MSSQL_sc.fit_transform(MSSQL_X_test)

#print data shapes
print(MSSQL_X_train.shape, MSSQL_X_test.shape)
print(MSSQL_y_train.shape, MSSQL_y_test.shape)

def run_model():
    #Train on MSSQL data
    clfr = RandomForestClassifier(n_estimators = 5)
    start_time = time.time()
    clfr.fit(MSSQL_X_train, MSSQL_y_train.values.ravel())
    end_time = time.time()
    print("Training time: ", end_time-start_time)

    #Test on Random Forest Model with MSSQL data
    start_time = time.time()
    MSSQL_y_test_pred = clfr.predict(MSSQL_X_test)
    end_time = time.time()
    print("Testing time: ", end_time-start_time)

    #Print MSSQL results
    print("Train score is:", clfr.score(MSSQL_X_train, MSSQL_y_train))
    print("Test score is:", clfr.score(MSSQL_X_test, MSSQL_y_test))
    #micro=Calculate metrics globally by counting the total true positives, false negatives and false positives.
    precision = precision_score(MSSQL_y_test, MSSQL_y_test_pred, average='macro')
    recall = recall_score(MSSQL_y_test, MSSQL_y_test_pred, average='macro')
    conf_matrix = confusion_matrix(MSSQL_y_test, MSSQL_y_test_pred)
    false_positive_rate = conf_matrix[0, 1] / (conf_matrix[0, 0] + conf_matrix[0, 1])
    print("Precision:", precision)
    print("Recall:", recall)
    print("False Positive Rate:", false_positive_rate)
    print("Confusion Matrix:")
    print(conf_matrix)

run_model()
