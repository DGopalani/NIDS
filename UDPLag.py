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

#Define the input CSV file path
UDPLag_csv_file_path = 'Data/01-12/UDPLag.csv'

UDPLag_df = pd.read_csv(UDPLag_csv_file_path)
# Select only numeric columns from the DataFrame
numeric_df = UDPLag_df.select_dtypes(include=['number'])

#df of categorical rows
# Filter the columns in df_subset that are not in numeric_df
categorical_columns = UDPLag_df.columns.difference(numeric_df.columns)
categorical_df = UDPLag_df[categorical_columns]

#apply feature mappin on label column
UDPLag_df[' Label'] = UDPLag_df[' Label'].map({'BENIGN': 0, 'UDP-lag': 1, 'UDP': 2, 'Syn': 3, 'WebDDoS': 4, 'DrDoS_MSSQL': 5})
#make column filter list based on dataset info
UDPLag_vars_to_keep = [' ACK Flag Count', 'Init_Win_bytes_forward', ' min_seg_size_forward', ' Fwd IAT Mean', ' Fwd IAT Max', ' Label']
#filter df based on filter list defined above
UDPLag_df_filtered = UDPLag_df[UDPLag_vars_to_keep]

#UDPLag test dataset
UDPLag_test_csv_file_path = 'Data/03-11/UDPLag.csv'
UDPLag_test_df = pd.read_csv(UDPLag_test_csv_file_path)
#apply feature mappin on label column
UDPLag_test_df[' Label'] = UDPLag_test_df[' Label'].map({'BENIGN': 0, 'UDPLag': 1, 'UDP': 2, 'Syn': 3, 'WebDDoS': 4, 'MSSQL': 5})
#rename wrongly named column for consistency
UDPLag_test_df.rename(columns={'_bInit_Winytes_forward': 'Init_Win_bytes_forward'}, inplace=True)
#make testing filtered df based on vars_to_keep
UDPLag_test_df_filtered = UDPLag_test_df[UDPLag_vars_to_keep]
#keep only 0, 1, and 4 in Label column of test_df_filtered
UDPLag_test_df_filtered = UDPLag_test_df_filtered[UDPLag_test_df_filtered[' Label'].isin([0, 1, 4])]

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

print("UDPLag")
#Prepare UDPLag train data
UDPLag_y_train = UDPLag_df_filtered[[' Label']]
UDPLag_X_train = UDPLag_df_filtered.drop(columns=[' Label'], axis=1)
UDPLag_sc = MinMaxScaler() 
UDPLag_X_train = UDPLag_sc.fit_transform(UDPLag_X_train)

#Prepare UDPLag test data
UDPLag_y_test = UDPLag_test_df_filtered[[' Label']]
UDPLag_X_test = UDPLag_test_df_filtered.drop(columns=[' Label'], axis=1)
UDPLag_sc = MinMaxScaler()
UDPLag_X_test = UDPLag_sc.fit_transform(UDPLag_X_test)

#print data shapes
print(UDPLag_X_train.shape, UDPLag_X_test.shape) 
print(UDPLag_y_train.shape, UDPLag_y_test.shape) 

def run_model():
    #Train on UDPLag data
    clfr = RandomForestClassifier(n_estimators = 5) 
    start_time = time.time() 
    clfr.fit(UDPLag_X_train, UDPLag_y_train.values.ravel()) 
    end_time = time.time() 
    print("Training time: ", end_time-start_time) 

    #Test on Random Forest Model with UDPLag data
    start_time = time.time() 
    y_test_pred = clfr.predict(UDPLag_X_test) 
    end_time = time.time() 
    print("Testing time: ", end_time-start_time) 

    #Print UDPLag results
    print("Train score is:", clfr.score(UDPLag_X_train, UDPLag_y_train)) 
    print("Test score is:", clfr.score(UDPLag_X_test, UDPLag_y_test)) 
    #micro=Calculate metrics globally by counting the total true positives, false negatives and false positives.
    #macro=Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
    precision = precision_score(UDPLag_y_test, y_test_pred, average='macro')
    recall = recall_score(UDPLag_y_test, y_test_pred, average='macro')
    conf_matrix = confusion_matrix(UDPLag_y_test, y_test_pred)
    false_positive_rate = conf_matrix[0, 1] / (conf_matrix[0, 0] + conf_matrix[0, 1])
    print("Precision:", precision)
    print("Recall:", recall)
    print("False Positive Rate:", false_positive_rate)
    print("Confusion Matrix:")
    print(conf_matrix)
