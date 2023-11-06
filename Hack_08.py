# Importing the required libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas_datareader as pdr
# Set the start and end dates for the data
start_date = '1960-01-01'
end_date   = '2020-01-01'
# Fetch S&P 500 price data
data = np.array((pdr.get_data_fred('SP500', start = start_date, 
                                            end = end_date)).dropna())
# Define the VAE model
latent_dim = 1
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
num_samples = 1
# Train the VAE on the normalized data using 10 epochs
history = vae.fit(normalized_data, normalized_data, epochs = 10, 
                  verbose = 1)
# Generate the synthetic data points
random_latent_vectors = np.random.normal(size=(num_samples, latent_dim))
synthetic_data_10 = decoder.predict(random_latent_vectors)
synthetic_data_10 = synthetic_data_10 * (np.max(data) - np.min(data))+\
                 np.min(data)
# Train the VAE on the normalized data  using 100 epochs
history = vae.fit(normalized_data, normalized_data, epochs = 100, 
                  verbose = 1)
random_latent_vectors = np.random.normal(size=(num_samples, latent_dim))
synthetic_data_100 = decoder.predict(random_latent_vectors)
synthetic_data_100 = synthetic_data_100 * (np.max(data) - np.min(data))+\
                 np.min(data)                               
# Train the VAE on the normalized data  using 1000 epochs
history = vae.fit(normalized_data, normalized_data, epochs = 1000, 
                  verbose = 1)
random_latent_vectors = np.random.normal(size=(num_samples, latent_dim))
synthetic_data_1000 = decoder.predict(random_latent_vectors)
synthetic_data_1000 = synthetic_data_1000 * (np.max(data) - np.min(data)) + np.min(data)                  
# Train the VAE on the normalized data  using 10000 epochs
history = vae.fit(normalized_data, normalized_data, epochs = 10000, 
                  verbose = 1)
random_latent_vectors = np.random.normal(size=(num_samples, latent_dim))
synthetic_data_10000 = decoder.predict(random_latent_vectors)
synthetic_data_10000 = synthetic_data_10000 * (np.max(data) - np.min(data)) + np.min(data)                 
synthetic_data_10 = np.transpose(synthetic_data_10)
synthetic_data_100 = np.transpose(synthetic_data_100)
synthetic_data_1000 = np.transpose(synthetic_data_1000)
synthetic_data_10000 = np.transpose(synthetic_data_10000)              
# Plot the original and synthetic data
fig, axs = plt.subplots(2, 1)
axs[0].plot(data, label = 'Original Data', color = 'black')
axs[0].legend()
axs[0].grid()
axs[1].plot(synthetic_data_10, label = 'Generated Data | 10 Epochs',
            color = 'blue')
axs[1].plot(synthetic_data_100, label = 'Generated Data | 100 Epochs',
            color = 'green')
axs[1].plot(synthetic_data_1000, label = 'Generated Data | 1000 Epochs',
            color = 'red')
axs[1].plot(synthetic_data_10000, label = 'Generated Data | 10000 Epochs',
            color = 'orange')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.legend()
plt.title('Original (black) vs. Synthetic Time Series Data (colored)')
plt.show()
plt.grid()