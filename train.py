import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import os
from scipy.io import wavfile
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from glob import glob
import argparse
import warnings


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, wav_paths, labels, sr, dt, n_classes, batch_size=32, shuffle=True):
        super().__init__()
        self.wav_paths = wav_paths
        self.labels = labels
        self.sr = sr
        self.dt = dt
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.wav_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        wav_paths = [self.wav_paths[k] for k in indexes]
        labels = [self.labels[k] for k in indexes]

        X = np.empty((self.batch_size, int(self.sr * self.dt), 1), dtype=np.float32)
        Y = np.empty((self.batch_size, self.n_classes), dtype=np.float32)

        for i, (path, label) in enumerate(zip(wav_paths, labels)):
            rate, wav = wavfile.read(path)
            wav = wav[:int(self.sr * self.dt)]
            if len(wav) < int(self.sr * self.dt):
                wav = np.pad(wav, (0, int(self.sr * self.dt) - len(wav)))
            X[i, ] = wav.reshape(-1, 1)
            Y[i, ] = to_categorical(label, num_classes=self.n_classes)

        # Debugging print statement
        print(f"Returning batch - X shape: {X.shape}, Y shape: {Y.shape}")

        return X, Y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.wav_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)



def Conv1D(N_CLASSES, SR, DT):
    input_shape = (int(SR * DT), 1)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(N_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def Conv2D(N_CLASSES, SR, DT):
    input_shape = (128, int(SR * DT / 128), 1)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(N_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def LSTM(N_CLASSES, SR, DT):
    input_shape = (int(SR * DT), 1)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(N_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def SimpleModel(N_CLASSES, SR, DT):
    input_shape = (int(SR * DT), 1)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(N_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train(args):
    src_root = args.src_root
    sr = args.sample_rate
    dt = args.delta_time
    batch_size = args.batch_size
    model_type = args.model_type

    params = {
        'N_CLASSES': len(os.listdir(src_root)),
        'SR': sr,
        'DT': dt
    }

    print("Params being passed to model creation:", params)

    model_creators = {
        'conv1d': Conv1D,
        'conv2d': Conv2D,
        'lstm': LSTM
    }

    assert model_type in model_creators.keys(), f'{model_type} not an available model'

    # Initialize the model with parameters
    model = model_creators[model_type](**params)

    # Verify the model architecture before proceeding to fitting
    print(model.summary())

    csv_path = os.path.join('logs', f'{model_type}_history.csv')
    os.makedirs('logs', exist_ok=True)

    wav_paths = glob(f'{src_root}/**/*.wav', recursive=True)

    if len(wav_paths) == 0:
        raise FileNotFoundError("No .wav files found in the specified directory.")

    # Limit dataset size
    if len(wav_paths) > 100:
        wav_paths = wav_paths[:100]

    print(f'Number of WAV files found (after limiting): {len(wav_paths)}')

    classes = sorted(os.listdir(src_root))
    print(f"Classes: {classes}")

    le = LabelEncoder()
    le.fit(classes)
    labels = [os.path.split(x)[0].split(os.sep)[-1] for x in wav_paths]
    labels = le.transform(labels)

    wav_train, wav_val, label_train, label_val = train_test_split(
        wav_paths, labels, test_size=0.1, random_state=0)

    if len(label_train) < batch_size:
        raise ValueError('Number of train samples must be >= batch_size')
    if len(set(label_train)) != params['N_CLASSES']:
        warnings.warn(f'Found {len(set(label_train))}/{params["N_CLASSES"]} classes in training data.')
    if len(set(label_val)) != params['N_CLASSES']:
        warnings.warn(f'Found {len(set(label_val))}/{params["N_CLASSES"]} classes in validation data.')

    print(f'Number of training samples: {len(wav_train)}')
    print(f'Number of validation samples: {len(wav_val)}')

    tg = DataGenerator(wav_train, label_train, sr, dt, params['N_CLASSES'], batch_size=batch_size)
    vg = DataGenerator(wav_val, label_val, sr, dt, params['N_CLASSES'], batch_size=batch_size)

    X_sample, Y_sample = tg[0]
    print(f"Sample batch shapes - X: {X_sample.shape}, Y: {Y_sample.shape}")

    os.makedirs('models', exist_ok=True)

    cp = ModelCheckpoint(f'models/{model_type}.keras', monitor='val_loss',
                         save_best_only=True, save_weights_only=False,
                         mode='auto', save_freq='epoch', verbose=1)

    csv_logger = CSVLogger(csv_path, append=False)

    # Train the model
    try:
        model.fit(tg, validation_data=vg, epochs=30, verbose=1, callbacks=[csv_logger, cp])
    except Exception as e:
        print(f"Error during model fitting: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Audio Classification Training')
    parser.add_argument('--model_type', type=str, default='lstm', help='model to run. i.e. conv1d, conv2d, lstm')
    parser.add_argument('--src_root', type=str, default='clean', help='directory of audio files in total duration')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--delta_time', '-dt', type=float, default=1.0, help='time in seconds to sample audio')
    parser.add_argument('--sample_rate', '-sr', type=int, default=16000, help='sample rate of clean audio')
    args = parser.parse_args()

    train(args)
