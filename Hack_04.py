# Importing the required libraries
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
# Generate a noisy sine wave as an example
np.random.seed(0)
x = np.linspace(0, 6 * np.pi, 1000)
y_clean = np.sin(x)
noise = np.random.normal(0, 1, x.shape)
y_noisy = y_clean + noise
# Create a DAE architecture for denoising
input_dim = 1
latent_dim = 1000
# Define the DAE model
inputs = Input(shape=(input_dim,))
# Encoder
enc = Dense(128, activation='relu')(inputs)
enc = Dense(128, activation='relu')(inputs) 
enc = Dense(128, activation='relu')(inputs) 
enc = Dense(128, activation='relu')(inputs)
enc = Dense(128, activation='relu')(inputs)
enc = Dense(latent_dim, activation='relu')(enc)
# Decoder
dec = Dense(128, activation='relu')(enc) 
dec = Dense(128, activation='relu')(enc)
dec = Dense(128, activation='relu')(enc)
dec = Dense(128, activation='relu')(enc)
dec = Dense(128, activation='relu')(enc)
dec = Dense(input_dim, activation='linear')(dec)
# Compile the DAE
dae = Model(inputs, dec)
dae.compile(optimizer='adam', loss='mean_squared_error')
# Train the DAE
dae.fit(y_noisy, y_clean, epochs = 100, batch_size = 8)
# Denoise the noisy sine wave
denoised_waveform = dae.predict(y_noisy)
# Plot the results
plt.figure()
plt.plot(x, y_clean, label = 'Clean Data', linewidth = 2.5)
plt.plot(x, y_noisy, label = 'Noisy Data', alpha = 0.7)
plt.plot(x, denoised_waveform, label = 'Denoised Data', color = 'r', 
         linestyle = '--', linewidth = 1)
plt.legend()
plt.title('Denoising Autoencoder Application')
plt.grid()
plt.show()