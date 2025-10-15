# Convert wav into numerical representation

import librosa
import numpy as np
from pathlib import Path
import config
import pretty_midi

HOP_LENGTH = 512
N_FFT = 2048

def audio_to_mel(wav_path):
    y, sr = librosa.load(wav_path, sr=config.SAMPLE_RATE)
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=config.N_MELS
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    # normalize to 0..1
    S_norm = (S_db + 80.0) / 80.0
    return S_norm.T.astype(np.float32)  # shape: (time_frames, n_mels)

def midi_to_events(midi_path):
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    events = []
    for inst in pm.instruments:
        for note in inst.notes:
            onset = round(note.start, 3)  # seconds
            dur = round(note.end - note.start, 3)
            events.append((onset, int(note.pitch), round(dur,3), int(note.velocity)))
    events.sort(key=lambda e: e[0])
    return events

# Tokenization: simple version
def events_to_tokens(events):
    tokens = []
    prev_onset = 0.0
    for onset, pitch, dur, vel in events:
        delta_ms = int(round((onset - prev_onset) * 1000))  # onset delta in ms
        dur_ms = int(round(dur * 1000))
        tokens.extend([f"DT_{delta_ms}", f"P_{pitch}", f"DU_{dur_ms}"])
        prev_onset = onset
    return tokens

def main():
    maestro_folder = Path(config.MAESTRO_PATH)
    
    # Pick a single example
    midi_file = next(maestro_folder.rglob("*.midi"))
    wav_file = midi_file.with_suffix(".wav")
    
    print("Processing audio:", wav_file)
    mel = audio_to_mel(wav_file)
    print("Mel-spectrogram shape:", mel.shape)

    midi_file = Path(config.MAESTRO_PATH) / "2018/MUS-2018_01/song1.midi"
    events = midi_to_events(midi_file)
    tokens = events_to_tokens(events)
    for token in tokens:
        print(token)

if __name__ == "__main__":
    main()
