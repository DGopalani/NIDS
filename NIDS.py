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
#import UDPLag
import UDP
#import SYN
import MSSQL

#Overall training dataset - contains UDP and UDPLag and SYN data
frames = [UDP.UDP_df, MSSQL.MSSQL_df]
#set vars to keep to UDP_vars_to_keep + UDPLag_vars_to_keep - Label
vars_to_keep = UDP.UDP_vars_to_keep + MSSQL.MSSQL_vars_to_keep[:-1]
vars_to_keep = list(set(vars_to_keep))
#filter df based on filter list defined above
df = pd.concat(frames)
df_filtered = df[vars_to_keep]

#print df_filtered label value counts
print(df_filtered[' Label'].value_counts())

#Overall test dataset
test_df = pd.read_csv('Data/03-11/UDP.csv')
#apply feature mapping on label column
test_df[' Label'] = test_df[' Label'].map({'BENIGN': 0, 'UDPLag': 1, 'UDP': 2, 'Syn': 3, 'WebDDoS': 4, 'MSSQL': 2})
#rename wrongly named column for consistency
#test_df.rename(columns={'_bInit_Winytes_forward': 'Init_Win_bytes_forward'}, inplace=True)
#make testing filtered df based on vars_to_keep
test_df_filtered = test_df[vars_to_keep]
#keep only 0, 1, and 4 in Label column of test_df_filtered
test_df_filtered = test_df_filtered[test_df_filtered[' Label'].isin([0, 2, 5])]

print(test_df_filtered[' Label'].value_counts())


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

#Setup UDPLag Model
#model_UDPLag = make_pipeline(MinMaxScaler(), RandomForestClassifier(n_estimators=5))
#model_UDPLag.fit(UDPLag.UDPLag_X_train, UDPLag.UDPLag_y_train.values.ravel())

#Setup UDP Model
model_UDP = make_pipeline(MinMaxScaler(), RandomForestClassifier(n_estimators=5))
model_UDP.fit(UDP.UDP_X_train, UDP.UDP_y_train.values.ravel())


#Setup MSSQL Model
model_MSSQL = make_pipeline(MinMaxScaler(), RandomForestClassifier(n_estimators=2))
model_MSSQL.fit(MSSQL.MSSQL_X_train, MSSQL.MSSQL_y_train.values.ravel())


#Setup SYN Model
#model_SYN = make_pipeline(MinMaxScaler(), RandomForestClassifier(n_estimators=5))
#model_SYN.fit(SYN.SYN_X_train, SYN.SYN_y_train.values.ravel())

print("UDP + MSSQL")

#prepare overall train data
y_train = df_filtered[[' Label']]
X_train = df_filtered.drop(columns=[' Label'], axis=1)
sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)

#Prepare overall test data
test_y = test_df_filtered[[' Label']]
test_X = test_df_filtered.drop(columns=[' Label'], axis=1)
#test_sc = MinMaxScaler()
test_X = sc.fit_transform(test_X)

print(X_train.shape, test_X.shape)
print(y_train.shape, test_y.shape)

# Combine models using a meta-model (Stacking)
combination = StackingClassifier(estimators=[
    ('UDP', model_UDP), 
    ('MSSQL', model_MSSQL)])
combination.fit(X_train, y_train.values.ravel())

# Test the meta-model on the combined test data
y_test_pred_combined = combination.predict(test_X)

print("Length of y_test_pred_combined:", len(y_test_pred_combined))
print("Length of test_y:", len(test_y))

# Evaluate the combined model
combined_accuracy = accuracy_score(test_y, y_test_pred_combined)
combined_precision = precision_score(test_y, y_test_pred_combined, average='macro')
combined_recall = recall_score(test_y, y_test_pred_combined, average='macro')
combined_conf_matrix = confusion_matrix(test_y, y_test_pred_combined)
false_positive_rate = combined_conf_matrix[0, 1] / (combined_conf_matrix[0, 0] + combined_conf_matrix[0, 1])

print("Combined Model Metrics:")
print("Combined Accuracy:", combined_accuracy)
print("Combined Precision:", combined_precision)
print("Combined Recall:", combined_recall)
print("Combined False Positive Rate:", false_positive_rate)
print("Combined Confusion Matrix:")
print(combined_conf_matrix)
