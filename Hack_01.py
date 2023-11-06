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