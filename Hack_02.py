# Importing the required libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
# Generate a synthetic dataset with anomalies
np.random.seed(0)
normal_data = np.random.normal(loc = 0, scale = 1, size = (5000, 10))
anomalies = np.random.uniform(low = -5, high = 5, size = (100, 10))
data = np.vstack((normal_data, anomalies))
# Create labels (0 for normal, 1 for anomalies)
labels = np.zeros(len(data))
labels[len(normal_data):] = 1
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, 
                                                    test_size = 0.2, 
                                                    random_state = 0)
# Define an Autoencoder model for anomaly detection
input_dim = X_train.shape[1]
model = keras.Sequential([
    keras.layers.Input(shape=(input_dim,)),
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dense(input_dim, activation = 'sigmoid')])
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
# Train the Autoencoder
model.fit(X_train, X_train, epochs = 50, batch_size = 16, 
          validation_data = (X_test, X_test))
# Calculate reconstruction errors on the test set
reconstructed_data = model.predict(X_test)
reconstruction_errors = np.mean(np.square(X_test - reconstructed_data), 
                                axis = 1)
# Define a threshold for anomaly detection using a 99th percentile
threshold = np.percentile(reconstruction_errors, 99)
# Classify data points as normal or anomalies based on the threshold
predicted_labels = (reconstruction_errors > threshold).astype(int)
# Evaluate the anomaly detection performance
accuracy = np.mean(predicted_labels == y_test)
precision = np.sum((predicted_labels == 1) & (y_test == 1)) / np.sum(predicted_labels == 1)
recall = np.sum((predicted_labels == 1) & (y_test == 1)) / np.sum(y_test == 1)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)