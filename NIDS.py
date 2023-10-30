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
csv_file_path = 'Data/03-11/UDPLag.csv' 

df = pd.read_csv(csv_file_path)

# Select only numeric columns from the DataFrame
numeric_df = df.select_dtypes(include=['number'])

'''
#df of categorical rows
# Filter the columns in df_subset that are not in numeric_df
categorical_columns = df.columns.difference(numeric_df.columns)
categorical_df = df[categorical_columns]
print(categorical_df.columns)

for column in categorical_df.columns:
    value_counts = categorical_df[column].value_counts()
    plt.figure(figsize=(8, 6))  # Adjust the figure size if needed
    # Plot the unique values on the y-axis and their counts on the x-axis
    value_counts.plot(kind='barh', color='skyblue')
    plt.xlabel('Count')
    plt.ylabel('Unique Values')
    plt.title(f'Count of Unique Values in {column}')
    plt.show()
'''

#numeric df stuff
correlation_matrix = numeric_df.corr()


# Create a heatmap to visualize the correlation matrix
'''
plt.figure(figsize=(8, 8))  # Adjust the figure size if needed
sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap (Numeric Data Only)")
plt.show()
'''

# Set a threshold for high correlation (e.g., 0.7)
threshold = 0.7

# Loop through the correlation matrix to identify high correlations
high_correlations = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            high_correlations.append((correlation_matrix.columns[i], correlation_matrix.columns[j]))


# Print the pairs of variables with high correlation
'''
for pair in high_correlations:
    var1, var2 = pair
    correlation_value = correlation_matrix.loc[var1, var2]
    print(f"High correlation between {var1} and {var2}: {correlation_value}")
'''

#keep only one of each pair of vars with high correlation
vars_to_keep = []
for pair in high_correlations:
    var1, var2 = pair
    if var1 not in vars_to_keep and var2 not in vars_to_keep:
        vars_to_keep.append(var1)

#add Label to vars_to_keep
vars_to_keep.append(' Label')

#apply feature mappin on label column
df[' Label'] = df[' Label'].map({'BENIGN': 0, 'UDPLag': 1, 'UDP': 2, 'Syn': 3})

#new filter list based on dataset info
vars_to_keep = [' ACK Flag Count', '_bInit_Winytes_forward', ' min_seg_size_forward', ' Fwd IAT Mean', ' Fwd IAT Max', ' Label']

#filter df based on vars_to_keep
df_filtered = df[vars_to_keep]

#Prepare train data
y_train = df_filtered[[' Label']]
X_train = df_filtered.drop(columns=[' Label'], axis=1)
sc = MinMaxScaler() 
X_train = sc.fit_transform(X_train)


#test dataset
test_csv_file_path = 'Data/01-12/UDPLag.csv'
test_df = pd.read_csv(test_csv_file_path)

#apply feature mappin on label column
print(test_df[' Label'].unique())
test_df[' Label'] = test_df[' Label'].map({'BENIGN': 0, 'UDP-lag': 1, 'WebDDoS': 2})

#test filtered df based on vars_to_keep
test_vars_to_keep = [' ACK Flag Count', 'Init_Win_bytes_forward', ' min_seg_size_forward', ' Fwd IAT Mean', ' Fwd IAT Max', ' Label']
test_df_filtered = test_df[test_vars_to_keep]

#Prepare test data
y_test = test_df_filtered[[' Label']]
X_test = test_df_filtered.drop(columns=[' Label'], axis=1)
sc = MinMaxScaler()
X_test = sc.fit_transform(X_test)

#print data shapes
print(X_train.shape, X_test.shape) 
print(y_train.shape, y_test.shape) 

#Train on Random Forest Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score 

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

