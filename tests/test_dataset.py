from src import config
from pathlib import Path

def main():
    maestro_folder = Path(config.MAESTRO_PATH)
    midi_files = list(maestro_folder.rglob("*.midi"))

    if not midi_files:
        print("No MIDI files found. Check your MAESTRO_PATH in .env.")
        return

    for midi_file in midi_files:
        audio_file = midi_file.with_suffix(".wav")
        print("Found MIDI:", midi_file)
        print("Corresponding audio exists?", audio_file.exists())

if __name__ == "__main__":
    main()
