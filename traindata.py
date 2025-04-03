import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

# Define dataset path
DATASET_PATH = 'wavfiles'  # Ensure your dataset follows the correct structure
data_dir = pathlib.Path(DATASET_PATH)

if not data_dir.exists():
    raise FileNotFoundError(f"Dataset directory '{DATASET_PATH}' not found. Ensure it is correctly placed.")

# Load dataset
train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=DATASET_PATH,
    batch_size=64,
    validation_split=0.2,
    seed=42,
    output_sequence_length=16000,
    subset='both')

# Extract class labels
label_names = np.array(train_ds.class_names)
print("Label names:", label_names)

# Function to squeeze extra dimensions
def squeeze(audio, labels):
    audio = tf.squeeze(audio, axis=-1)
    return audio, labels

# Apply squeezing transformation
train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)

# Split validation dataset into validation and test sets
test_ds = val_ds.shard(num_shards=2, index=0)
val_ds = val_ds.shard(num_shards=2, index=1)

# Visualizing example audio waveforms
for example_audio, example_labels in train_ds.take(1):  
    print("Example audio shape:", example_audio.shape)
    print("Example labels shape:", example_labels.shape)

    plt.figure(figsize=(16, 10))
    rows, cols = 3, 3
    n = rows * cols
    
    for i in range(n):
        if i >= len(example_audio):
            break
        plt.subplot(rows, cols, i+1)
        plt.plot(example_audio[i])
        plt.title(label_names[example_labels[i].numpy()])  # Fixed: Added .numpy()
        plt.yticks(np.arange(-1.2, 1.2, 0.2))
        plt.ylim([-1.1, 1.1])
    
    plt.show()


# Waveform to spectrogram
def get_spectrogram(waveform):
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(
        waveform, frame_length=255, frame_step=128)
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

# Visualize spectrogram for first 3 examples
for i in range(3):
    label = label_names[example_labels[i].numpy()]  # Fixed: Added .numpy()
    waveform = example_audio[i]
    spectrogram = get_spectrogram(waveform)

    print('Label:', label)
    print('Waveform shape:', waveform.shape)
    print('Spectrogram shape:', spectrogram.shape)
    print('Audio playback')
    display.display(display.Audio(waveform, rate=16000))

def plot_spectrogram(spectrogram, ax):
    if len(spectrogram.shape) > 2:
        assert len(spectrogram.shape) == 3
        spectrogram = np.squeeze(spectrogram, axis=-1)
    # Convert the frequencies to log scale and transpose, so that the time is
    # represented on the x-axis (columns).
    # Add an epsilon to avoid taking a log of zero.
    log_spec = np.log(spectrogram.T + np.finfo(float).eps)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)

# Plot spectrograms for the first batch
fig, axes = plt.subplots(2, figsize=(12, 8))
timescale = np.arange(waveform.shape[0])
axes[0].plot(timescale, waveform.numpy())
axes[0].set_title('Waveform')
axes[0].set_xlim([0, 16000])

plot_spectrogram(spectrogram.numpy(), axes[1])
axes[1].set_title('Spectrogram')
plt.suptitle(label.title())
plt.show()

# Function to create spectrogram dataset
def make_spec_ds(ds):
    return ds.map(
        map_func=lambda audio, label: (get_spectrogram(audio), label),
        num_parallel_calls=tf.data.AUTOTUNE)

train_spectrogram_ds = make_spec_ds(train_ds)
val_spectrogram_ds = make_spec_ds(val_ds)
test_spectrogram_ds = make_spec_ds(test_ds)

for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
    break

rows = 3
cols = 3
n = rows * cols
fig, axes = plt.subplots(rows, cols, figsize=(16, 9))

for i in range(n):
    r = i // cols
    c = i % cols
    ax = axes[r][c]
    plot_spectrogram(example_spectrograms[i].numpy(), ax)  # Fixed: Added .numpy()
    ax.set_title(label_names[example_spect_labels[i].numpy()])  # Fixed: Added .numpy()

plt.show()

# Dataset caching, shuffling, and prefetching
train_spectrogram_ds = train_spectrogram_ds.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
val_spectrogram_ds = val_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)
test_spectrogram_ds = test_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)

# Define model input shape and number of labels
input_shape = example_spectrograms.shape[1:]
print('Input shape:', input_shape)
num_labels = len(label_names)

# Instantiate the `tf.keras.layers.Normalization` layer.
norm_layer = layers.Normalization()
# Fit the state of the layer to the spectrograms
# with `Normalization.adapt`.
norm_layer.adapt(data=train_spectrogram_ds.map(map_func=lambda spec, label: spec))

# Define the model
model = models.Sequential([
    layers.Input(shape=input_shape),
    layers.Resizing(32, 32),
    norm_layer,
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_labels),
])

model.summary()

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

# Train the model
# Method to save the model (keep this one)
# Method to save the model
def save_model(model, model_name='my_model'):
    # Ensure the saved_model directory exists
    save_dir = './saved_model'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Save the model in the 'saved_model' format
    save_path = os.path.join(save_dir, model_name + '.keras')  # Added .keras extension
    model.save(save_path)
    print(f"Model saved to {save_path}")

# Train the model
EPOCHS = 10
history = model.fit(
    train_spectrogram_ds,
    validation_data=val_spectrogram_ds,
    epochs=EPOCHS,
    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
    validation_freq=1  # Explicitly set validation frequency
)

# Save the trained model (do this after training, no need to call fit() again)
save_model(model, model_name='fruit_ripeness_model')

metrics = history.history
plt.figure(figsize=(16, 6))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(history.epoch, metrics['loss'], label='loss')
if 'val_loss' in metrics:
    plt.plot(history.epoch, metrics['val_loss'], label='val_loss')
plt.legend()
plt.ylim([0, max(plt.ylim())])
plt.xlabel('Epoch')
plt.ylabel('Loss [CrossEntropy]')

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(history.epoch, 100 * np.array(metrics['accuracy']), label='accuracy')
if 'val_accuracy' in metrics:
    plt.plot(history.epoch, 100 * np.array(metrics['val_accuracy']), label='val_accuracy')
plt.legend()
plt.ylim([0, 100])
plt.xlabel('Epoch')
plt.ylabel('Accuracy [%]')

plt.show()

# Evaluate the model
model.evaluate(test_spectrogram_ds, return_dict=True)

# Predict on test dataset
y_pred = model.predict(test_spectrogram_ds)

# Convert predictions to class labels
y_pred = tf.argmax(y_pred, axis=1)

# Get true labels
y_true = tf.concat(list(test_spectrogram_ds.map(lambda s, lab: lab)), axis=0)

# Confusion matrix
confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx,
            xticklabels=label_names,
            yticklabels=label_names,
            annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()
