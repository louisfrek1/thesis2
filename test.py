from kapre.composed import get_melspectrogram_layer
from tensorflow.keras.layers import Input
import tensorflow as tf

# Input shape: (batch_size, time_steps, channels)
input_shape = (1, 16000, 1)
input_data = Input(shape=(16000, 1))

# Apply the Mel Spectrogram Layer
mel_layer = get_melspectrogram_layer(input_shape=input_shape)(input_data)
print(mel_layer)
