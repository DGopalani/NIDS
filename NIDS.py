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
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from treeinterpreter import treeinterpreter as ti
import UDPLag
import UDP
import SYN
import MSSQL


#Overall training dataset - contains UDP and UDPLag and SYN data
frames = [UDPLag.UDPLag_df, UDP.UDP_df, SYN.SYN_df]
#set vars to keep to UDP_vars_to_keep + UDPLag_vars_to_keep - Label
vars_to_keep = UDPLag.UDPLag_vars_to_keep + UDP.UDP_vars_to_keep + SYN.SYN_vars_to_keep
vars_to_keep = list(set(vars_to_keep))

print("vars_to_keep:", vars_to_keep)
#filter df based on filter list defined above
df = pd.concat(frames)
df_filtered = df[vars_to_keep]

#print df_filtered label value counts
print("Training Label Value Counts:")
print(df_filtered[' Label'].value_counts())

#Overall test dataset
test_df = pd.read_csv('Data/03-11/UDPLag.csv')
#apply feature mapping on label column
test_df[' Label'] = test_df[' Label'].map({'BENIGN': 0, 'UDPLag': 1, 'UDP': 2, 'Syn': 3, 'WebDDoS': 4, 'MSSQL': 5})
#rename wrongly named column for consistency
test_df.rename(columns={'_bInit_Winytes_forward': 'Init_Win_bytes_forward'}, inplace=True)
#make testing filtered df based on vars_to_keep


test_df_filtered = test_df[vars_to_keep]

vars_to_keep.append(' Timestamp')
vars_to_keep.append(' Source IP')
vars_to_keep.append(' Destination IP')
vars_to_keep.append(' Source Port')
vars_to_keep.append(' Destination Port')

test_df_filtered_to_print = test_df[vars_to_keep]

test_df_filtered = test_df_filtered[test_df_filtered[' Label'].isin([0, 1, 2, 3])]

#print test_df_filtered label value counts
print("Testing Label Value Counts:")
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
model_UDPLag = make_pipeline(MinMaxScaler(), RandomForestClassifier(n_estimators=5))
model_UDPLag.fit(UDPLag.UDPLag_X_train, UDPLag.UDPLag_y_train.values.ravel())

#Setup UDP Model
model_UDP = make_pipeline(MinMaxScaler(), RandomForestClassifier(n_estimators=5))
model_UDP.fit(UDP.UDP_X_train, UDP.UDP_y_train.values.ravel())

#Setup SYN Model
model_SYN = make_pipeline(MinMaxScaler(), RandomForestClassifier(n_estimators=5))
model_SYN.fit(SYN.SYN_X_train, SYN.SYN_y_train.values.ravel())

'''
#Setup MSSQL Model
model_MSSQL = make_pipeline(MinMaxScaler(), RandomForestClassifier(n_estimators=2))
model_MSSQL.fit(MSSQL.MSSQL_X_train, MSSQL.MSSQL_y_train.values.ravel())
'''

print("UDPLag + UDP + Syn Model Metrics:")

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
    ('UDPLag', model_UDPLag),
    ('SYN', model_SYN)])
combination.fit(X_train, y_train.values.ravel())

# Print feature importances for each base model
for name, model in combination.named_estimators_.items():
    if hasattr(model, 'feature_importances_'):
        print(f"Feature Importances for {name}:")
        for feature_name, importance in zip(vars_to_keep, model.feature_importances_):
            print(f"Feature: {feature_name}, Importance: {importance}")
        print("\n")

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

print("\n")

#Hold feature descriptions
feature_descriptions = {}
# Add all feature descriptions to a dictionary
feature_descriptions[' Fwd Packet Length Std'] = "Standard deviation of forward packet lengths."
feature_descriptions['Init_Win_bytes_forward'] = "Initial size of the TCP window (in bytes) for the forward direction of the communication. The TCP window size is an important parameter in controlling the flow of data between two communicating devices."
feature_descriptions[' Protocol'] = "network protocol used in the communication, such as TCP, UDP, or other protocols."
feature_descriptions[' Flow Duration'] = "Total duration of the flow, i.e., the time elapsed from the start to the end of the communication flow."
feature_descriptions[' Fwd IAT Max'] = "Maximum inter-arrival time between forward packets in a flow. It gives the maximum time gap observed between consecutive forward data packets."
feature_descriptions[' ACK Flag Count'] = "Number of acknowledgment (ACK) flags in the network traffic. ACK flags are part of the TCP protocol and are used to acknowledge the receipt of data."
feature_descriptions[' Fwd IAT Mean'] = "Represents the mean (average) inter-arrival time between forward data packets in a flow. It provides a measure of the typical time gap between consecutive forward packets."
feature_descriptions[' min_seg_size_forward'] = "Minimum segment size observed in the forward direction. In TCP communication, the segment size refers to the amount of data that can be sent in a single TCP segment."
feature_descriptions['Fwd IAT Total'] = "Fwd IAT (Inter-Arrival Time) Total is the total time between two consecutive forward data packets in a flow. It provides insights into the time intervals between packets."
feature_descriptions[' Destination Port'] = "Indicates destination port number in the network traffic. Port numbers help identify specific services or applications running on a device. "
feature_descriptions[' Packet Length Std'] = "Standard deviation of packet lengths in the network traffic. It provides a measure of the variability or dispersion of packet sizes in the flow."
feature_descriptions[' Label'] = "Resultant label of the network traffic, such as benign, UDP, SYN..."

UDP_description = {}
UDP_description[' Destination Port'] = "In UDP flood attacks, attackers may target specific ports associated with vulnerable services or flood random ports to exhaust network resources. Destination port can determine if it is UDP."
UDP_description[' Fwd Packet Length Std'] = "Unusually high or low standard deviations of packet length might indicate anomalies or malicious patterns in the UDP traffic."
UDP_description[' Packet Length Std'] = "Gives a broader view of the variability of packet lengths in the entire communication flow."
UDP_description[' min_seg_size_forward'] = "Anomalous segment sizes may be used by attackers to manipulate or exploit vulnerabilities in the target system."
UDP_description[' Protocol'] = "Protocol feature is used to see if targeting UDP protocol"

UDPLag_description = {}
UDPLag_description[' ACK Flag Count'] = "Even though this is attack uses a UDP connectionless protocol, which doesnt involve the ACK flag used in the TCP handshake,it can still be useful when ​​looking at other protocols or if there's a need to distinguish UDP packets from other types of traffic."
UDPLag_description['Init_Win_bytes_forward'] = "Initial size of the TCP window (in bytes) for the forward direction of the communication. The TCP window size is an important parameter in controlling the flow of data between two communicating devices."
UDPLag_description[' min_seg_size_forward'] = "Anomalous segment sizes may be used by attackers to manipulate or exploit vulnerabilities in the target system."
UDPLag_description[' Fwd IAT Mean'] = 'Monitoring the average inter-arrival time helps identify patterns associated with intentionally spacing out UDP packets to cause lag.'
UDPLag_description[' FWD IAT Max'] = 'Relevant in detecting instances where the time gap between UDP packets is maximized to amplify the impact on network latency.'

#Print likelihood of each class
print("Likelihood of each class:")
probabilities = combination.predict_proba(test_X)

# Print the probabilities for the first few samples
for i in range(min(100, len(test_X))):
    print(f"Test Instance {i + 1}:")

    #Print selected class
    predicted_class = ""
    selected_class_features = []
    if y_test_pred_combined[i] == 0:
        predicted_class = "BENIGN"
        selected_class_features = vars_to_keep
    elif y_test_pred_combined[i] == 1:
        predicted_class = "UDPLag"
        selected_class_features = UDPLag.UDPLag_vars_to_keep
    elif y_test_pred_combined[i] == 2:
        predicted_class = "UDP"
        selected_class_features = UDP.UDP_vars_to_keep
    elif y_test_pred_combined[i] == 3:
        predicted_class = "SYN"
        selected_class_features = SYN.SYN_vars_to_keep
    elif y_test_pred_combined[i] == 4:
        predicted_class = "WebDDoS"
        selected_class_features = UDPLag.UDPLag_vars_to_keep

    print(f"Predicted Class: {predicted_class}")

    #Print feature values for this test instance
    #print values of whole row of this test instance:
    print("Feature Values:")
    print(test_df_filtered_to_print.iloc[i])
    print("Relevant Feature Descriptions: ")

    # Print the feature values for the selected class and description of feature
    for feature in selected_class_features:
        if feature in feature_descriptions:
            print(f"Feature: {feature}, Value: {test_df_filtered_to_print[feature].iloc[i]}, Description: {feature_descriptions[feature]}")

    # Print predicted probabilities for each class
    print("Predicted Probabilities:")
    for j, prob in enumerate(probabilities[i]):
        if j == 0:
            print(f"Class {j} (BENIGN): {prob:.4f}")
        elif j == 1:
            print(f"Class {j} (UDPLag): {prob:.4f}")
        elif j == 2:
            print(f"Class {j} (UDP): {prob:.4f}")
        elif j == 3:
            print(f"Class {j} (SYN): {prob:.4f}")
        elif j == 4:
            print(f"Class {j} (WebDDoS): {prob:.4f}")
    
    print("\n")
