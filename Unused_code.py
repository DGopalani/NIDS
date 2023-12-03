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
'''