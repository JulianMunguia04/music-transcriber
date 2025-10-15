# src/config.py

from dotenv import load_dotenv
import os

# Load variables from .env
load_dotenv()

# Paths
MAESTRO_PATH = os.getenv("MAESTRO_PATH")
MUSESCORE_PATH = os.getenv("MUSESCORE_PATH")
OUTPUT_PATH = os.getenv("OUTPUT_PATH")

# Audio settings
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", 22050))
N_MELS = int(os.getenv("N_MELS", 80))

# Optional: print to verify
if __name__ == "__main__":
    print("MAESTRO_PATH:", MAESTRO_PATH)
    print("MUSESCORE_PATH:", MUSESCORE_PATH)
    print("OUTPUT_PATH:", OUTPUT_PATH)
    print("SAMPLE_RATE:", SAMPLE_RATE)
    print("N_MELS:", N_MELS)
