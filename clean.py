import matplotlib.pyplot as plt
from scipy.io import wavfile
import argparse
import os
from glob import glob
import numpy as np
import pandas as pd
from librosa.core import resample, to_mono
from tqdm import tqdm
import wavio
import librosa
from models import Conv1D, Conv2D, LSTM


def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate / 20), min_periods=1, center=True).max()
    for mean in y_mean:
        mask.append(mean > threshold)
    return mask, y_mean


def downsample_mono(path, sr):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")
    
    obj = wavio.read(path)
    wav = obj.data.astype(np.float32, order='F')
    rate = obj.rate
    
    try:
        if wav.ndim > 1:  # Check for multiple channels
            wav = to_mono(wav.T)
    except Exception as exc:
        raise exc
    
    wav = resample(wav, orig_sr=rate, target_sr=sr)
    wav = wav.astype(np.int16)
    return sr, wav


def save_sample(sample, rate, target_dir, fn, ix):
    fn = os.path.splitext(fn)[0]
    dst_path = os.path.join(target_dir, f"{fn}_{ix}.wav")
    
    if os.path.exists(dst_path):
        return  # Skip if the file already exists
    
    wavfile.write(dst_path, rate, sample)


def check_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def split_wavs(args):
    src_root = args.src_root
    dst_root = args.dst_root
    dt = args.delta_time

    if not os.path.exists(src_root):
        raise FileNotFoundError(f"Source directory not found: {src_root}")

    wav_paths = glob(f'{src_root}/**/*.wav', recursive=True)
    if not wav_paths:
        raise FileNotFoundError(f"No .wav files found in directory: {src_root}")

    print(f"Found {len(wav_paths)} .wav files in {src_root}")

    check_dir(dst_root)

    classes = [d for d in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, d))]
    for _cls in classes:
        target_dir = os.path.join(dst_root, _cls)
        check_dir(target_dir)
        src_dir = os.path.join(src_root, _cls)
        for fn in tqdm(os.listdir(src_dir), desc=f"Processing {_cls}"):
            src_fn = os.path.join(src_dir, fn)
            rate, wav = downsample_mono(src_fn, args.sr)
            mask, y_mean = envelope(wav, rate, threshold=args.threshold)
            wav = wav[mask]
            delta_sample = int(dt * rate)

            if wav.shape[0] < delta_sample:
                sample = np.zeros(shape=(delta_sample,), dtype=np.int16)
                sample[:wav.shape[0]] = wav
                save_sample(sample, rate, target_dir, fn, 0)
            else:
                trunc = wav.shape[0] % delta_sample
                for cnt, i in enumerate(np.arange(0, wav.shape[0] - trunc, delta_sample)):
                    start = int(i)
                    stop = int(i + delta_sample)
                    sample = wav[start:stop]
                    save_sample(sample, rate, target_dir, fn, cnt)


def test_threshold(args):
    src_root = args.src_root
    wav_paths = glob(f'{src_root}/**', recursive=True)
    wav_paths = [x for x in wav_paths if '.wav' in x]
    
    wav_path = [x for x in wav_paths if args.fn in x]
    
    if len(wav_path) != 1:
        print(f'Audio file not found for substring: {args.fn}')
        if not wav_paths:
            print(f"No `.wav` files found in the source directory: {src_root}")
        else:
            print(f"Available `.wav` files in `{src_root}`:")
            for file in wav_paths:
                print(f"  - {os.path.basename(file)}")
        return
    
    print(f'Found file: {wav_path[0]}')
    rate, wav = downsample_mono(wav_path[0], args.sr)
    mask, env = envelope(wav, rate, threshold=args.threshold)
    plt.style.use('ggplot')
    plt.title(f'Signal Envelope, Threshold = {args.threshold}')
    plt.plot(wav[np.logical_not(mask)], color='r', label='remove')
    plt.plot(wav[mask], color='c', label='keep')
    plt.plot(env, color='m', label='envelope')
    plt.grid(False)
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cleaning audio data')
    parser.add_argument('--src_root', type=str, default='wavfiles',
                        help='Directory of audio files.')
    parser.add_argument('--dst_root', type=str, default='clean',
                        help='Directory to save cleaned audio files.')
    parser.add_argument('--delta_time', '-dt', type=float, default=1.0,
                        help='Time in seconds to sample audio.')
    parser.add_argument('--sr', type=int, default=16000,
                        help='Target sampling rate.')
    parser.add_argument('--fn', type=str, default='',
                        help='Substring of the filename to test.')
    parser.add_argument('--threshold', type=int, default=20,
                        help='Threshold for magnitude filtering.')

    args, _ = parser.parse_known_args()

    if args.fn:
        test_threshold(args)
    else:
        split_wavs(args)
