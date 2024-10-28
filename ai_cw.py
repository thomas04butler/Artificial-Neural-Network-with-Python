import math
from typing import Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import mplcursors

# Load data
df = pd.read_excel('/Users/thomasbutler/Desktop/AI CW/FEHDataStudent.xlsx')

# Analyze the data
mean_value = df['Index flood'].mean()
median_value = df['Index flood'].median()
mode_value = df['Index flood'].mode().iloc[0]
data_range = df['Index flood'].max() - df['Index flood'].min()
variance_value = df['Index flood'].var()
std_dev_value = df['Index flood'].std()

summary_stats = df['Index flood'].describe()

correlation_matrix = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap')
plt.show()

# Split the data
X = df.drop(columns=['Index flood'])
y = df['Index flood']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

X_test = pd.DataFrame(X_test)

pd.set_option('display.float_format', '{:.4f}'.format)

# Standardize the features
min_custom = 0.1
max_custom = 0.9

scaler = StandardScaler()

min_max_scaler = MinMaxScaler(feature_range=(min_custom, max_custom))
scaler_target = StandardScaler()

y_train_reshaped = y_train.values.reshape(-1, 1)

scaler_target.fit(y_train_reshaped)

y_train_standardized = scaler_target.transform(y_train_reshaped)
y_val_standardized = scaler_target.transform(y_val.values.reshape(-1, 1))

y_train_scaled_custom = min_max_scaler.fit_transform(y_train_standardized)
y_val_scaled_custom = min_max_scaler.transform(y_val_standardized)

X_train_standardized = scaler.fit_transform(X_train)
X_val_standardized = scaler.transform(X_val)

X_train_scaled_custom = min_max_scaler.fit_transform(X_train_standardized)
X_val_scaled_custom = min_max_scaler.transform(X_val_standardized)

#Functions

#Create Sigmoid Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#Calculate weighted sum 
def weightedSum(selected_features, weightValues, biasList):
    weighted_sum = np.dot(selected_features, weightValues[:, 1:].T) + biasList[:-1]  
    return weighted_sum

#Calculate Activation using sigmoid and weighted sum
def activationCalculate(weighted_sum):
    activation = sigmoid(weighted_sum)
    return activation

#Calucate mean loss
def loss(predicted_output, actual_output):
    mse_loss = np.mean((predicted_output - actual_output) ** 2)
    return mse_loss

#Update Weights and Biases

def weight_bias_update(weightValues, learning_rate, selected_features, biasList, activation, selected_IF):
    #Calculate delta
    delta = (activation - selected_IF) * activation * (1 - activation)
    #Update Weights
    delta_weights = -learning_rate * np.dot(delta[:, np.newaxis], selected_features[np.newaxis, :])
    weightValues[:, 1:] += delta_weights  
    #Update Biases 
    delta_bias = -learning_rate * delta
    biasList[:-1] += delta_bias
    return weightValues, biasList

#Loop for doing forward and bacwkward pass for a single epoch

def MLP_iteration(X_train, y_train, weightValues, biasList, learning_rate, num_epochs):
    errors = []  
    for _ in range(num_epochs):
        epoch_error = 0  
        for i in range(len(X_train)):
            #Get row of data in training set
            selected_features = X_train[i]
            selected_IF = y_train[i]

            #Forward Pass
            result_WeightedSum = weightedSum(selected_features, weightValues, biasList)
            activation = activationCalculate(result_WeightedSum)

            #Backward Pass
            loss_value = loss(activation, selected_IF)
            epoch_error += loss_value  
            weightValues, biasList = weight_bias_update(weightValues, learning_rate, selected_features, biasList, activation, selected_IF)
        errors.append(epoch_error / len(X_train))  

    return errors

#Changing learning rates 

num_epochs = 100
learning_rates = [0.15, 0.1, 0.05, 0.01, 0.005, 0.001,0.0005,0.0001,0.00001,0.000001]

plt.figure(figsize=(15, 10))

min_errors = [] 

numberOfHiddenNodes = 8
numberOfBiases = numberOfHiddenNodes + 1
numberOfWeights = numberOfHiddenNodes * (X_train.shape[1] + 1)
weightValues = np.random.uniform(-0.25, 0.25, (numberOfHiddenNodes, X_train.shape[1] + 1))
biasList = np.random.uniform(-0.25, 0.25, numberOfBiases)

for learning_rate in learning_rates:
    errors = MLP_iteration(X_train_scaled_custom, y_train_scaled_custom, weightValues, biasList, learning_rate, num_epochs)
    min_error = min(errors)
    min_errors.append(min_error)

plt.plot(learning_rates, min_errors, marker='o')
plt.xlabel('Learning Rate')
plt.ylabel('Minimum Mean Squared Error')
plt.title('Minimum Error vs. Learning Rate')
plt.show()

# Creating a separate figure for the table
plt.figure(figsize=(10, 5))

# Create a table to display learning rates and min errors
table_values = [[lr, err] for lr, err in zip(learning_rates, min_errors)]
table_headers = ['Learning Rate', 'Minimum Error']
table = plt.table(cellText=table_values, colLabels=table_headers, loc='center')

plt.axis('off')  # Hide axis for the table
plt.show()

#Changing amount of hidden nodes

learning_rate = 0.1
num_epochs = 100

hidden_nodes_range = range(4, 17)
mean_squared_errors = []  # List to store mean squared error at epoch 100

plt.figure(figsize=(10, 6))

handles = []

for numberOfHiddenNodes in hidden_nodes_range:
    
    numberOfBiases = numberOfHiddenNodes + 1
    numberOfWeights = numberOfHiddenNodes * (X_train.shape[1] + 1)
    weightValues = np.random.uniform(-0.25, 0.25, (numberOfHiddenNodes, X_train.shape[1] + 1))
    biasList = np.random.uniform(-0.25, 0.25, numberOfBiases)
    errors = MLP_iteration(X_train_scaled_custom, y_train_scaled_custom, weightValues, biasList, learning_rate, num_epochs)
    
    mean_squared_errors.append(errors[99])
    
    handle, = plt.plot(range(1, num_epochs + 1), errors, label=f'Hidden Nodes: {numberOfHiddenNodes}')
    handles.append(handle)

plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('Error Rate per Epoch')
plt.legend()

mplcursors.cursor(handles)

plt.show()

# Plot the mean squared error at epoch 100 for each number of hidden nodes
plt.figure(figsize=(8, 5))
plt.plot(hidden_nodes_range, mean_squared_errors, marker='o', linestyle='-')
plt.xlabel('Number of Hidden Nodes')
plt.ylabel('Mean Squared Error at Epoch 100')
plt.title('Mean Squared Error at Epoch 100 vs. Number of Hidden Nodes')
plt.grid(True)
plt.show()

#Changing amount of epochs

import numpy as np
import matplotlib.pyplot as plt

learning_rate = 0.0001
num_epochs_list = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 10000]

plt.figure(figsize=(15, 10))

min_errors = []

numberOfHiddenNodes = 8

for num_epochs in num_epochs_list:
    numberOfBiases = numberOfHiddenNodes + 1
    numberOfWeights = numberOfHiddenNodes * (X_train.shape[1] + 1)
    weightValues = np.random.uniform(-0.25, 0.25, (numberOfHiddenNodes, X_train.shape[1] + 1))
    biasList = np.random.uniform(-0.25, 0.25, numberOfBiases)
    errors = MLP_iteration(X_train_scaled_custom, y_train_scaled_custom, weightValues, biasList, learning_rate, num_epochs)

    min_error = min(errors)
    min_errors.append(min_error)

plt.plot(num_epochs_list, min_errors, marker='o')
plt.xlabel('Number of Epochs')
plt.ylabel('Minimum Mean Squared Error')
plt.title('Minimum Error vs. Number of Epochs')
plt.show()

plt.figure(figsize=(8, 5))
plt.table(cellText=np.array([min_errors]).T, rowLabels=num_epochs_list, colLabels=["Minimum MSE"], loc='center')
plt.axis('off')
plt.title('Minimum Mean Squared Error for Different Number of Epochs')
plt.show()
