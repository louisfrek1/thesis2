from tensorflow.keras.models import load_model
from clean import downsample_mono, envelope
from kapre.time_frequency import STFT, Magnitude, ApplyFilterbank, MagnitudeToDecibel
from sklearn.preprocessing import LabelEncoder
import numpy as np
from glob import glob
import argparse
import os
import pandas as pd
from tqdm import tqdm

def make_prediction(args):
    # Check if the model file exists before loading
    if not os.path.exists(args.model_fn):
        raise FileNotFoundError(f"Model file '{args.model_fn}' not found. Ensure the correct path.")

    # Load the trained model
    model = load_model(args.model_fn, custom_objects={
        'STFT': STFT,
        'Magnitude': Magnitude,
        'ApplyFilterbank': ApplyFilterbank,
        'MagnitudeToDecibel': MagnitudeToDecibel
    })
    
    # Get WAV file paths
    wav_paths = glob(os.path.join(args.src_dir, '**', '*.wav'), recursive=True)
    wav_paths = sorted([x.replace(os.sep, '/') for x in wav_paths])  # Normalize paths
    
    if not wav_paths:
        raise FileNotFoundError(f"No .wav files found in '{args.src_dir}'. Check the directory path.")

    # Prepare label encoding
    classes = sorted(os.listdir(args.src_dir))
    le = LabelEncoder()
    y_true = le.fit_transform([os.path.basename(os.path.dirname(x)) for x in wav_paths])

    results = []
    for wav_fn in tqdm(wav_paths, desc="Processing Audio Files"):
        # Process the WAV file
        rate, wav = downsample_mono(wav_fn, args.sr)
        mask, env = envelope(wav, rate, threshold=args.threshold)
        clean_wav = wav[mask]

        step = int(args.sr * args.dt)
        batch = []

        # Slice audio into fixed-size samples
        for i in range(0, clean_wav.shape[0], step):
            sample = clean_wav[i:i+step]
            sample = sample.reshape(-1, 1)

            if sample.shape[0] < step:  # Pad shorter samples
                tmp = np.zeros((step, 1), dtype=np.float32)
                tmp[:sample.shape[0], :] = sample
                sample = tmp

            batch.append(sample)

        # Convert batch to numpy array and make predictions
        X_batch = np.array(batch, dtype=np.float32)
        y_pred = model.predict(X_batch)
        y_mean = np.mean(y_pred, axis=0)
        y_pred_label = np.argmax(y_mean)

        real_class = os.path.basename(os.path.dirname(wav_fn))
        predicted_class = classes[y_pred_label]

        print(f'Actual class: {real_class}, Predicted class: {predicted_class}')
        results.append(y_mean)

    # Save predictions
    os.makedirs('logs', exist_ok=True)
    np.save(os.path.join('logs', args.pred_fn), np.array(results))
    print(f"Predictions saved to 'logs/{args.pred_fn}.npy'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Audio Classification Prediction')
    parser.add_argument('--model_fn', type=str, default='models/lstm.keras',
                        help='Path to the trained model file')
    parser.add_argument('--pred_fn', type=str, default='y_pred',
                        help='Filename for saving predictions in logs directory')
    parser.add_argument('--src_dir', type=str, default='wavfiles',
                        help='Directory containing WAV files for prediction')
    parser.add_argument('--dt', type=float, default=1.0,
                        help='Duration in seconds to sample audio')
    parser.add_argument('--sr', type=int, default=16000,
                        help='Sample rate of the audio')
    parser.add_argument('--threshold', type=int, default=20,
                        help='Threshold magnitude for np.int16 dtype')

    args = parser.parse_args()
    make_prediction(args)
