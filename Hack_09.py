# Importing the required libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas_datareader as pdr
# Set the start and end dates for the data
start_date = '2000-01-01'
end_date   = '2020-01-01'
# Fetch S&P 500 price data
eurusd = np.reshape(np.array((pdr.get_data_fred('DEXUSEU', start = start_date, 
                                            end = end_date)).dropna()), 
                                            (-1))

usdchf = np.reshape(np.array((pdr.get_data_fred('DEXSZUS', start = start_date, 
                                            end = end_date)).dropna()), 
                                            (-1))
print('Correlation = ', round(np.corrcoef(eurusd, usdchf)[0, 1], 2))
# Define the VAE model for EURUSD
latent_dim = 1
num_samples = 1
encoder = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(eurusd),)),  
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
    tf.keras.layers.Dense(len(eurusd), activation='linear'),
])
vae = tf.keras.Model(encoder.inputs, decoder(encoder.outputs))
# Compile the model
vae.compile(optimizer = 'adam', loss = 'mse')
# Normalize the data
normalized_eurusd = (eurusd - np.min(eurusd)) / (np.max(eurusd) - 
                    np.min(eurusd))
normalized_eurusd = normalized_eurusd.reshape(1, -1)
# Train the VAE on the normalized data using 10000 epochs
history = vae.fit(normalized_eurusd, normalized_eurusd, epochs = 10000, 
                  verbose = 1)
# Generate the synthetic data points
random_latent_vectors = np.random.normal(size=(num_samples, latent_dim))
synthetic_data_eurusd = decoder.predict(random_latent_vectors)
synthetic_data_eurusd = synthetic_data_eurusd * (np.max(eurusd) - 
                 np.min(eurusd)) + np.min(eurusd)
# Define the VAE model for USDCHF
encoder = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(usdchf),)),  
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
    tf.keras.layers.Dense(len(usdchf), activation='linear'),
])
vae = tf.keras.Model(encoder.inputs, decoder(encoder.outputs))
# Compile the model
vae.compile(optimizer = 'adam', loss = 'mse')
# Normalize the data
normalized_usdchf = (usdchf - np.min(usdchf)) / (np.max(usdchf) - 
                    np.min(usdchf))
normalized_usdchf = normalized_usdchf.reshape(1, -1)
# Train the VAE on the normalized data using 10000 epochs
history = vae.fit(normalized_usdchf, normalized_usdchf, epochs = 10000, 
                  verbose = 1)
# Generate the synthetic data points
random_latent_vectors = np.random.normal(size=(num_samples, latent_dim))
synthetic_data_usdchf = decoder.predict(random_latent_vectors)
synthetic_data_usdchf = synthetic_data_usdchf * (np.max(usdchf) - 
                        np.min(usdchf)) + np.min(usdchf)
synthetic_data_eurusd = np.reshape(np.transpose(synthetic_data_eurusd), 
                        (-1))
synthetic_data_usdchf = np.reshape(np.transpose(synthetic_data_usdchf), 
                        (-1))
print('Correlation = ', round(np.corrcoef(synthetic_data_eurusd, 
      synthetic_data_usdchf)[0, 1], 2))
# Plot the original and synthetic data
fig, axs = plt.subplots(2, 1)
axs[0].plot(synthetic_data_eurusd, label = 'Synthetic EURUSD', 
            color = 'black')
axs[0].legend()
axs[0].grid()
axs[1].plot(synthetic_data_usdchf, label = 'Synthetic USDCHF', 
            color = 'blue')
axs[1].legend()
axs[1].grid()