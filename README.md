# 🎵 Maestro Dataset Project

This project uses the MAESTRO dataset for piano music processing and analysis.  
Follow the instructions below to set up your environment and project structure.

---

Folder Structure

project_root/

│

├── data/

│   └── maestro-v3.0.0/           # Place the extracted MAESTRO dataset here
│

├── notebooks/                    # Jupyter notebooks for analysis

│

├── src/                          # Source code (models, utilities, etc.)

│

├── .env                          # Environment variables (see below)

│

├── requirements.txt              # Python dependencies

│

└── README.md

---

Environment Setup

1. Create and activate a virtual environment

python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

2. Install dependencies

pip install -r requirements.txt

3. Create a .env file in the project root

# Example .env
MAESTRO_PATH="C:/Users/Julian/Datasets/maestro-v3.0.0"
MUSESCORE_PATH="C:/Program Files/MuseScore 4/bin/MuseScore4.exe"
OUTPUT_PATH="./data/outputs"
SAMPLE_RATE=22050
N_MELS=80

---

Usage

1. Preprocess the dataset
python src/preprocess.py

2. Train the model
python src/train.py

3. Run inference
python src/infer.py path/to/song.mp3

4. Convert MusicXML → MuseScore (.mscz)
python src/convert.py --input data/outputs/output.musicxml

---

Notes

- MAESTRO provides aligned audio + MIDI pairs — no manual synchronization needed.  
- The model learns to predict sequences of notes (pitch, onset, duration) and reconstructs them as MusicXML.  
- MuseScore is used for converting and visualizing MusicXML output.
