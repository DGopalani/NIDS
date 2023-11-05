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

#UDP training dataset
UDP_csv_file_path = 'Data/01-12/DrDoS_UDP.csv'
UDP_df = pd.read_csv(UDP_csv_file_path)

'''
#make bar graphs of categorical data
for column in categorical_df.columns:
    value_counts = categorical_df[column].value_counts()
    plt.figure(figsize=(8, 6))  # Adjust the figure size if needed
    # Plot the unique values on the y-axis and their counts on the x-axis
    value_counts.plot(kind='barh', color='skyblue')
    plt.xlabel('Count')
    plt.ylabel('Unique Values')
    plt.title(f'Count of Unique Values in {column}')
    plt.show()

#make correlation matrix
correlation_matrix = numeric_df.corr()

# Create a heatmap to visualize the correlation matrix
plt.figure(figsize=(8, 8))  # Adjust the figure size if needed
sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap (Numeric Data Only)")
plt.show()

# Set a threshold for high correlation (e.g., 0.7)
threshold = 0.7

# Loop through the correlation matrix to identify high correlations
high_correlations = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            high_correlations.append((correlation_matrix.columns[i], correlation_matrix.columns[j]))


# Print the pairs of variables with high correlation
for pair in high_correlations:
    var1, var2 = pair
    correlation_value = correlation_matrix.loc[var1, var2]
    print(f"High correlation between {var1} and {var2}: {correlation_value}")

#keep only one of each pair of vars with high correlation
vars_to_keep = []
for pair in high_correlations:
    var1, var2 = pair
    if var1 not in vars_to_keep and var2 not in vars_to_keep:
        vars_to_keep.append(var1)
#add Label to vars_to_keep
vars_to_keep.append(' Label')
'''

#apply feature mappin on label column
UDPLag_df[' Label'] = UDPLag_df[' Label'].map({'BENIGN': 0, 'UDP-lag': 1, 'UDP': 2, 'Syn': 3, 'WebDDoS': 4})
UDP_df[' Label'] = UDP_df[' Label'].map({'BENIGN': 0, 'UDP-lag': 1, 'UDP': 2, 'Syn': 3, 'WebDDoS': 4})

#make column filter list based on dataset info
UDPLag_vars_to_keep = [' ACK Flag Count', 'Init_Win_bytes_forward', ' min_seg_size_forward', ' Fwd IAT Mean', ' Fwd IAT Max', ' Label']
UDP_vars_to_keep = [' ACK Flag Count', 'Init_Win_bytes_forward', ' min_seg_size_forward', ' Fwd IAT Mean', ' Fwd IAT Max', ' Label']

#filter df based on filter list defined above
UDPLag_df_filtered = UDPLag_df[UDPLag_vars_to_keep]

UPD_df_filtered = UDP_df[UDP_vars_to_keep]


#test dataset
test_csv_file_path = 'Data/03-11/UDPLag.csv'
test_df = pd.read_csv(test_csv_file_path)

#apply feature mappin on label column
test_df[' Label'] = test_df[' Label'].map({'BENIGN': 0, 'UDPLag': 1, 'UDP': 2, 'Syn': 3, 'WebDDoS': 4})
#rename wrongly named column for consistency
test_df.rename(columns={'_bInit_Winytes_forward': 'Init_Win_bytes_forward'}, inplace=True)

#make testing filtered df based on vars_to_keep
test_df_filtered = test_df[UDPLag_vars_to_keep]

#keep only 0, 1, and 4 in Label column of test_df_filtered
test_df_filtered = test_df_filtered[test_df_filtered[' Label'].isin([0, 1, 4])]

#Prepare train data
y_train = UDPLag_df_filtered[[' Label']]
X_train = UDPLag_df_filtered.drop(columns=[' Label'], axis=1)
sc = MinMaxScaler() 
X_train = sc.fit_transform(X_train)

#Prepare test data
y_test = test_df_filtered[[' Label']]
X_test = test_df_filtered.drop(columns=[' Label'], axis=1)
sc = MinMaxScaler()
X_test = sc.fit_transform(X_test)


#check for null values
print(np.isnan(y_test.values.any()))
print(np.isfinite(y_test.values.all()))
 
#print data shapes
print(X_train.shape, X_test.shape) 
print(y_train.shape, y_test.shape) 

#Train on Random Forest Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("Random Forest Classifier")
clfr = RandomForestClassifier(n_estimators = 5) 
start_time = time.time() 
clfr.fit(X_train, y_train.values.ravel()) 
end_time = time.time() 
print("Training time: ", end_time-start_time) 

#Test on Random Forest Model
start_time = time.time() 
y_test_pred = clfr.predict(X_train) 
end_time = time.time() 
print("Testing time: ", end_time-start_time) 

#Print results
print("Train score is:", clfr.score(X_train, y_train)) 
print("Test score is:", clfr.score(X_test, y_test)) 
'''
print("Gaussian Naive Bayes Classifier")
# Gaussian Naive Bayes 
from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import accuracy_score 
  
clfg = GaussianNB() 
start_time = time.time() 
clfg.fit(X_train, y_train.values.ravel()) 
end_time = time.time() 
print("Training time: ", end_time-start_time) 

start_time = time.time() 
y_test_pred = clfg.predict(X_train) 
end_time = time.time() 
print("Testing time: ", end_time-start_time) 

print("Train score is:", clfg.score(X_train, y_train)) 
print("Test score is:", clfg.score(X_test, y_test)) 
'''

# Decision Tree 
print("Decision Tree Classifier") 
from sklearn.tree import DecisionTreeClassifier 
clfd = DecisionTreeClassifier(criterion ="entropy", max_depth = 4) 
start_time = time.time() 
clfd.fit(X_train, y_train.values.ravel()) 
end_time = time.time() 
print("Training time: ", end_time-start_time) 
start_time = time.time() 
y_test_pred = clfd.predict(X_train) 
end_time = time.time() 
print("Testing time: ", end_time-start_time) 
print("Train score is:", clfd.score(X_train, y_train)) 
print("Test score is:", clfd.score(X_test, y_test)) 
