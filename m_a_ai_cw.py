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

#Update Weights and Biases with momentum

def weight_bias_update_with_momentum(weightValues, prev_delta_weights, learning_rate, selected_features, biasList, activation, selected_IF, momentum):
    delta = (activation - selected_IF) * activation * (1 - activation)
    delta_weights = -learning_rate * np.dot(delta[:, np.newaxis], selected_features[np.newaxis, :]) + momentum * prev_delta_weights
    delta_bias = -learning_rate * delta
    weightValues[:, 1:] += delta_weights
    biasList[:-1] += delta_bias
    return weightValues, biasList, delta_weights

#Loop for doing forward and bacwkward pass for a single epoch

def MLP_iteration_with_momentum(X_train, y_train, weightValues, biasList, learning_rate, num_epochs, momentum):
    errors = []
    prev_delta_weights = np.zeros_like(weightValues[:, 1:])
    
    for _ in range(num_epochs):
        epoch_error = 0
        for i in range(len(X_train)):
            selected_features = X_train[i]
            selected_IF = y_train[i]
            result_WeightedSum = weightedSum(selected_features, weightValues, biasList)
            activation = activationCalculate(result_WeightedSum)
            loss_value = loss(activation, selected_IF)
            epoch_error += loss_value
            weightValues, biasList, prev_delta_weights = weight_bias_update_with_momentum(weightValues, prev_delta_weights, learning_rate, selected_features, biasList, activation, selected_IF, momentum)
        errors.append(epoch_error / len(X_train))
    
    return errors

# Initialize weights and biases
numberOfHiddenNodes = 8
numberOfBiases = numberOfHiddenNodes + 1
numberOfWeights = numberOfHiddenNodes * (X_train.shape[1] + 1)
weightValues = np.random.uniform(-0.25, 0.25, (numberOfHiddenNodes, X_train.shape[1] + 1))
biasList = np.random.uniform(-0.25, 0.25, numberOfBiases)

learning_rate = 0.1
num_epochs = 100
momentum = 0.9

# Training with momentum
errors_with_momentum = MLP_iteration_with_momentum(X_train_scaled_custom, y_train_scaled_custom, weightValues.copy(), biasList.copy(), learning_rate, num_epochs, momentum)
min_error_with_momentum = min(errors_with_momentum)

print("Minimum Mean Squared Error with momentum:", min_error_with_momentum)

# Training without momentum
errors_without_momentum = MLP_iteration_with_momentum(X_train_scaled_custom, y_train_scaled_custom, weightValues.copy(), biasList.copy(), learning_rate, num_epochs, momentum=0)
min_error_without_momentum = min(errors_without_momentum)

print("Minimum Mean Squared Error without momentum:", min_error_without_momentum)

plt.figure(figsize=(10, 6))

plt.plot(range(1, num_epochs + 1), errors_with_momentum, label="With Momentum")
plt.plot(range(1, num_epochs + 1), errors_without_momentum, label="Without Momentum")
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.title('Mean Squared Error vs. Epochs')
plt.legend()
plt.grid(True)
plt.show()

def annealed_learning_rate(initial_learning_rate, epoch, annealing_rate):
    return initial_learning_rate / (1 + annealing_rate * epoch)

def MLP_iteration_with_annealing(X_train, y_train, weightValues, biasList, initial_learning_rate, num_epochs, momentum, annealing_rate):
    errors = []
    prev_delta_weights = np.zeros_like(weightValues[:, 1:])
    
    for epoch in range(num_epochs):
        epoch_error = 0
        learning_rate = annealed_learning_rate(initial_learning_rate, epoch, annealing_rate)
        
        for i in range(len(X_train)):
            selected_features = X_train[i]
            selected_IF = y_train[i]
            result_WeightedSum = weightedSum(selected_features, weightValues, biasList)
            activation = activationCalculate(result_WeightedSum)
            loss_value = loss(activation, selected_IF)
            epoch_error += loss_value
            weightValues, biasList, prev_delta_weights = weight_bias_update_with_momentum(weightValues, prev_delta_weights, learning_rate, selected_features, biasList, activation, selected_IF, momentum)
        errors.append(epoch_error / len(X_train))
    
    return errors

# Initialize parameters
learning_rate = 0.1
num_epochs = 100
momentum = 0.9
annealing_rate = 0.001

# Training with annealing
errors_with_annealing = MLP_iteration_with_annealing(X_train_scaled_custom, y_train_scaled_custom, weightValues.copy(), biasList.copy(), learning_rate, num_epochs, momentum, annealing_rate)
min_error_with_annealing = min(errors_with_annealing)

print("Minimum Mean Squared Error with annealing:", min_error_with_annealing)

plt.figure(figsize=(10, 6))

plt.plot(range(1, num_epochs + 1), errors_with_momentum, label="With Momentum")
plt.plot(range(1, num_epochs + 1), errors_with_annealing, label="With Annealing")
plt.plot(range(1, num_epochs + 1), errors_without_momentum, label="Without Momentum and annealing")
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.title('Mean Squared Error vs. Epochs')
plt.legend()
plt.grid(True)
plt.show()
