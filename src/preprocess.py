import librosa
import numpy as np
from pathlib import Path
from src import config
import pretty_midi
import music21
from music21 import note, chord, stream, tempo, duration

HOP_LENGTH = 512
N_FFT = 2048

def tokens_to_events(tokens):
    """Convert tokens back to event sequence"""
    events = []
    for token in tokens:
        if token.startswith('P_'):
            events.append(('NOTE_ON', int(token[2:])))
        elif token.startswith('DT_'):
            events.append(('TIME_SHIFT', int(token[3:]) / 1000.0))  # Convert back to seconds
        elif token.startswith('DU_'):
            events.append(('DURATION', int(token[3:]) / 1000.0))  # Convert back to seconds
    return events

def events_to_midi(events, output_path, tempo_bpm=120):
    """Convert events back to MIDI file"""
    # Create a music21 stream
    s = stream.Stream()
    s.insert(0, tempo.MetronomeMark(number=tempo_bpm))
    
    current_time = 0.0
    active_notes = {}  # pitch -> (start_time, duration)
    
    for event_type, value in events:
        if event_type == 'NOTE_ON':
            # Start a new note
            active_notes[value] = (current_time, None)
            
        elif event_type == 'DURATION':
            # Update the duration for the most recent note
            if active_notes:
                # Get the most recently started note
                latest_pitch = list(active_notes.keys())[-1]
                start_time, _ = active_notes[latest_pitch]
                active_notes[latest_pitch] = (start_time, value)
                
        elif event_type == 'TIME_SHIFT':
            # Finalize notes that end during this time shift
            notes_to_remove = []
            for pitch, (start_time, note_duration) in active_notes.items():
                if note_duration is not None and start_time + note_duration <= current_time + value:
                    # Note ends during this time shift, create the note
                    n = note.Note(pitch)
                    n.offset = start_time
                    n.duration = duration.Duration(note_duration)
                    s.insert(start_time, n)
                    notes_to_remove.append(pitch)
            
            # Remove finalized notes
            for pitch in notes_to_remove:
                del active_notes[pitch]
            
            current_time += value
    
    # Finalize any remaining notes
    for pitch, (start_time, note_duration) in active_notes.items():
        if note_duration is not None:
            n = note.Note(pitch)
            n.offset = start_time
            n.duration = duration.Duration(note_duration)
            s.insert(start_time, n)
    
    # Save to MIDI file
    s.write('midi', fp=output_path)
    print(f"MIDI file saved: {output_path}")
    return s

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

def midi_to_event_sequence(midi_path):
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

def build_example(wav_path, midi_path):
    """
    Converts WAV + MIDI paths into:
    X = mel-spectrogram (numpy array)
    tokens = list of token strings for decoder
    """
    X = audio_to_mel(wav_path)
    events = midi_to_event_sequence(midi_path)
    tokens = events_to_tokens(events)
    return X, tokens


def main():
    maestro_folder = Path(config.MAESTRO_PATH)
    
    # Pick a single example
    midi_file = next(maestro_folder.rglob("*.midi"))
    wav_file = midi_file.with_suffix(".wav")
    
    print("Processing audio:", wav_file)
    mel = audio_to_mel(wav_file)
    print("Mel-spectrogram shape:", mel.shape)

    midi_file = Path(config.MAESTRO_PATH) / "2018/MUS-2018_01/song1.midi"
    events = midi_to_event_sequence(midi_file)
    tokens = events_to_tokens(events)
    for token in tokens:
        print(token)

if __name__ == "__main__":
    main()
