import librosa
import numpy as np

def preprocess_audio(audio_path, target_sr=16000):
    audio, sr = librosa.load(audio_path, sr=None)
    print(f"Original Sample Rate: {sr}Hz")

    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        print(f"Resampled from {sr} to {target_sr}Hz")

    audio = audio / np.max(np.abs(audio))
    print(f"Normalized")

    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=target_sr,
        n_mels=80
    )
    print(f"Mel Spectrogram Shape: {mel_spec.shape}")

    return audio, mel_spec


if __name__ == "__main__":
    from data_loader import load_dataset

    files = load_dataset()
    first_file = files[0]
    print(f"\nProcessing: {first_file['id']}")
    audio, mel = preprocess_audio(first_file['path'])
    print(f"\nAudio ready for Whisper!")