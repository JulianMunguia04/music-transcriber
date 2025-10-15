from src import config
from src.preprocess import audio_to_mel
from pathlib import Path
import matplotlib.pyplot as plt

def main():
    maestro_folder = Path(config.MAESTRO_PATH)
    midi_files = list(maestro_folder.rglob("*.midi"))

    if not midi_files:
        print("No MIDI files found. Check your MAESTRO_PATH in .env.")
        return

    for midi_file in midi_files[:1]:  # just pick the first one for testing
        audio_file = midi_file.with_suffix(".wav")
        print("Found MIDI:", midi_file)
        print("Corresponding audio exists?", audio_file.exists())

        if not audio_file.exists():
            continue

        # Convert to mel-spectrogram
        mel = audio_to_mel(audio_file)
        print("Mel-spectrogram shape:", mel.shape)
        print("First 5 time frames:\n", mel[:5])

        # Visualize
        plt.figure(figsize=(10, 4))
        plt.imshow(mel.T, aspect='auto', origin='lower', cmap='magma')
        plt.xlabel("Time frames")
        plt.ylabel("Mel bins")
        plt.title(f"Mel-Spectrogram for {audio_file.name}")
        plt.colorbar(label="Normalized amplitude")
        plt.show()

if __name__ == "__main__":
    main()
