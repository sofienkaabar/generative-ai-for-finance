# Importing the required libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas_datareader as pdr
# Set the start and end dates for the data
start_date = '1960-01-01'
end_date   = '2023-09-01'
# Import the data
data = pdr.DataReader('CPIAUCSL', 'fred', start_date, end_date).dropna()
data = data.pct_change(periods = 12, axis = 0) * 100
data = data.dropna()
data = np.array(data)
# Define the VAE model
latent_dim = 2
encoder = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(data),)),  
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),    
    tf.keras.layers.Dense(latent_dim, activation='relu'),
])
decoder = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(latent_dim,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),    
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
                  verbose = 0)
# Generate 5 synthetic data points
num_samples = 5
random_latent_vectors = np.random.normal(size=(num_samples, latent_dim))
synthetic_data = decoder.predict(random_latent_vectors)
# Denormalize the synthetic data
from statsmodels.tsa.stattools import adfuller
synthetic_data = np.transpose(synthetic_data)
print('p-value: %f' % adfuller(data)[1])
print('p-value: %f' % adfuller(synthetic_data[:, 0])[1])
print('p-value: %f' % adfuller(synthetic_data[:, 1])[1])
print('p-value: %f' % adfuller(synthetic_data[:, 2])[1])
print('p-value: %f' % adfuller(synthetic_data[:, 3])[1])
print('p-value: %f' % adfuller(synthetic_data[:, 4])[1])