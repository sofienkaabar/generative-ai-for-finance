# Importing the required libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
# Fetch S&P 500 price data
data = np.reshape(np.array(pd.read_excel('ISM_PMI.xlsx')), (-1))
data = np.diff(data)
# Define the VAE model
latent_dim = 1
encoder = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(data),)),  
    tf.keras.layers.Dense(500, activation='relu'),
    tf.keras.layers.Dense(500, activation='relu'),
    tf.keras.layers.Dense(500, activation='relu'),    
    tf.keras.layers.Dense(latent_dim, activation='relu'),
])
decoder = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(latent_dim,)),
    tf.keras.layers.Dense(500, activation='relu'),
    tf.keras.layers.Dense(500, activation='relu'),
    tf.keras.layers.Dense(500, activation='relu'),    
    tf.keras.layers.Dense(len(data), activation='linear'),
])
vae = tf.keras.Model(encoder.inputs, decoder(encoder.outputs))
# Compile the model
vae.compile(optimizer = 'adam', loss = 'mse')
# Normalize the data
normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
normalized_data = normalized_data.reshape(1, -1)
# Train the VAE on the normalized data
history = vae.fit(normalized_data, normalized_data, epochs = 1000, 
                  verbose = 1)
# Generate 5 synthetic data points
num_samples = 5
random_latent_vectors = np.random.normal(size=(num_samples, latent_dim))
synthetic_data = decoder.predict(random_latent_vectors)
# Denormalize the synthetic data
synthetic_data = synthetic_data * (np.max(data) - np.min(data)) + \
                 np.min(data)                
# Plot the original and synthetic data
fig, axs = plt.subplots(2, 1)
axs[0].plot(data, label = 'Original Data', color = 'black')
axs[0].legend()
axs[0].grid()
for i in range(num_samples):
    plt.plot(synthetic_data[i], label = f'Synthetic Data {i+1}', 
             linewidth = 1)
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.legend()
plt.title('Original (black) vs. Synthetic Time Series Data (colored)')
plt.show()
plt.grid()
# Importing linear regresstion
from sklearn.linear_model import LinearRegression
# Data preprocessing function
def data_preprocessing(data, num_lags, train_test_split):
    x = []
    y = []
    for i in range(len(data) - num_lags):
        x.append(data[i:i + num_lags])
        y.append(data[i+ num_lags])
    x = np.array(x)
    y = np.array(y)
    split_index = int(train_test_split * len(x))
    x_train = x[:split_index]
    y_train = y[:split_index]
    x_test = x[split_index:]
    y_test = y[split_index:]
    return x_train, y_train, x_test, y_test
# Transposing
synthetic_data = np.transpose(synthetic_data)
# Preparing the training and test sets
x_train_original, y_train_original, x_test_original, y_test_original = data_preprocessing(data, 80, 0.80)
x_train_simulated_1, y_train_simulated_1, x_test_simulated_1, y_test_simulated_1  = data_preprocessing(synthetic_data[:, 0], 80, 0.80)
x_train_simulated_2, y_train_simulated_2, x_test_simulated_2, y_test_simulated_2  = data_preprocessing(synthetic_data[:, 1], 80, 0.80)
x_train_simulated_3, y_train_simulated_3, x_test_simulated_3, y_test_simulated_3  = data_preprocessing(synthetic_data[:, 2], 80, 0.80)
x_train_simulated_4, y_train_simulated_4, x_test_simulated_4, y_test_simulated_4  = data_preprocessing(synthetic_data[:, 3], 80, 0.80)
x_train_simulated_5, y_train_simulated_5, x_test_simulated_5, y_test_simulated_5  = data_preprocessing(synthetic_data[:, 4], 80, 0.80)
# Fitting the models
model_original = LinearRegression()
model_simulated_1 = LinearRegression()
model_simulated_2 = LinearRegression()
model_simulated_3 = LinearRegression()
model_simulated_4 = LinearRegression()
model_simulated_5 = LinearRegression()
model_original.fit(x_train_original, y_train_original)
model_simulated_1.fit(x_train_simulated_1, y_train_simulated_1)
model_simulated_2.fit(x_train_simulated_2, y_train_simulated_2)
model_simulated_3.fit(x_train_simulated_3, y_train_simulated_3)
model_simulated_4.fit(x_train_simulated_4, y_train_simulated_4)
model_simulated_5.fit(x_train_simulated_5, y_train_simulated_5)
# Predicting out-of-sample
y_predicted_original    = np.reshape(model_original.predict
                                               (x_test_original), (-1))
y_predicted_simulated_1 = np.reshape(model_simulated_1.predict
                                               (x_test_simulated_1), (-1))
y_predicted_simulated_2 = np.reshape(model_simulated_2.predict
                                               (x_test_simulated_2), (-1))
y_predicted_simulated_3 = np.reshape(model_simulated_3.predict
                                               (x_test_simulated_3), (-1))
y_predicted_simulated_4 = np.reshape(model_simulated_4.predict
                                              (x_test_simulated_4), (-1))
y_predicted_simulated_5 = np.reshape(model_simulated_5.predict
                                              (x_test_simulated_5), (-1))
# Evaluation
same_sign_count_original = np.sum(np.sign(y_predicted_original) == 
            np.sign(y_test_original)) / len(y_test_original) * 100
same_sign_count_simulated_1 = np.sum(np.sign(y_predicted_simulated_1) == 
            np.sign(y_test_simulated_1)) / len(y_test_simulated_1) * 100
same_sign_count_simulated_2 = np.sum(np.sign(y_predicted_simulated_2) == 
            np.sign(y_test_simulated_2)) / len(y_test_simulated_2) * 100
same_sign_count_simulated_3 = np.sum(np.sign(y_predicted_simulated_3) == 
            np.sign(y_test_simulated_3)) / len(y_test_simulated_3) * 100
same_sign_count_simulated_4 = np.sum(np.sign(y_predicted_simulated_4) == 
            np.sign(y_test_simulated_4)) / len(y_test_simulated_4) * 100
same_sign_count_simulated_5 = np.sum(np.sign(y_predicted_simulated_5) == 
            np.sign(y_test_simulated_5)) / len(y_test_simulated_5) * 100
print('Hit Ratio Original Data = ', same_sign_count_original, '%')
print('Hit Ratio Simulated Data 1 = ', same_sign_count_simulated_1, '%')
print('Hit Ratio Simulated Data 2 = ', same_sign_count_simulated_2, '%')
print('Hit Ratio Simulated Data 3 = ', same_sign_count_simulated_3, '%')
print('Hit Ratio Simulated Data 4 = ', same_sign_count_simulated_4, '%')
print('Hit Ratio Simulated Data 5 = ', same_sign_count_simulated_5, '%')