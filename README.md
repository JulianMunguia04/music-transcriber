# 🎵 Music Transcriber - Audio to MID

A scalable transformer-based deep learning model that converts WAV audio files into precise MIDI sequences. Train on the MAESTRO dataset with progressive scaling (small to large models) and generate professional-quality musical notation from any audio input. Perfect for musicians, producers, and developers looking to bridge the gap between recorded audio and digital music production.

---

## 🧰 Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white&style=for-the-badge)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white&style=for-the-badge)
![Keras](https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=white&style=for-the-badge)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white&style=for-the-badge)
![LibROSA](https://img.shields.io/badge/LibROSA-0B99B7?logo=librosa&logoColor=white&style=for-the-badge)
![MIDI](https://img.shields.io/badge/MIDI-000000?logo=midi&logoColor=white&style=for-the-badge)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white&style=for-the-badge)
![Git](https://img.shields.io/badge/Git-F05032?logo=git&logoColor=white&style=for-the-badge)

---

## Local Setup 🛠️

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/music-transcriber.git
cd music-transcriber
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
> This Project uses `Python 3.11`

### 3. Download MAESTRO Dataset
The model is trained on Google's MAESTRO dataset, which is a combination of `.wav` files and their respective `.midi` interpretations.
1. Visit [MAESTRO DATASET](https://magenta.withgoogle.com/datasets/maestro#dataset)
2. Download the [dataset](https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip) (≈120GB)
3. Extract to /data/maestro/

### 4. Build Vocabulary
```bash
python build_full_vocab.py
```
This processes the dataset and creates token mappings.

### 5. Configure Enviroment
Create a `.env` file for configuration:
```env
MAESTRO_PATH="C:/Users/Julian/Desktop/music_transcriber/data/maestro"
MAESTRO_PATH_FULL= "C:/Users/Julian/Desktop/music_transcriber/data/maestro-v3.0.0"
MUSESCORE_PATH="C:\Program Files\MuseScore 4\bin\MuseScore4.exe"
OUTPUT_PATH="./data/outputs"
SAMPLE_RATE=22050
N_MELS=80
```

##### Folder Structure

```
project_root/
│
├── data/
│   └── maestro-v3.0.0/  # Place the extracted MAESTRO dataset here
│
├── notebooks/           # Jupyter notebooks for analysis
│
├── src/                 # Source code (models, utilities, etc.)
│
├── wav_to_midi.py       # Use model to convert audios to mid
│
├── .env                 # Environment variables (see below)
│
├── requirements.txt     # Python dependencies
│
└── README.md
```

## Training the Model 🚀 

### Start with small scale (Recommended)
```python
# In train_scalable.py, set:
scale = 'small'
```
> The `scale` value can be changed to trained different sized models. Select from `"small"`, `"medium"`, `"large"` and `"full"`
###### Train

```bash
python -m src.train_scalable
```
> In `root` directory

### Progressive Training
1. Small (2-4 hours): 1% data, small model (~15M parameters)
2. Medium (8-12 hours): 10% data, medium model (~50M parameters)
3. Large (24+ hours): 50% data, large model (~200M parameters)
4. Full (48+ hours): 100% data, large model (~200M parameters)

#### Modify the `train_scalable.py`:
```python
scale = # or 'large' or 'full'
```

---

## Converting Audio to MID 🎵
### Basic Usage
```bash
python wav_to_midi.py path/to/your_audio.wav
```

### Advanced Options
```bash
python wav_to_midi.py input.wav \
  --output my_song.mid \
  --model-weights scalable_model_medium.weights.h5 \
  --max-length 1000 \
  --temperature 0.8
```
> Change scalable_model_medium.weights.h5 from `medium` to `small`, `large`, `full`.

### Arguments
* `--output, -o`: Output MIDI file path
* `--model-weights, -m`: Model weights file (default: scalable_model_small.weights.h5)
* `--max-length, -l`: Maximum sequence length (default: 1000)
* `--temperature, -t`: Sampling temperature (default: 0.8)

---

## Troubleshooting 🔧

### Out of Memory Erros
* Reduce `batch_size` in training config
* Decrease `MAX_SEQUENCE_LENGTH`
* Use smaller scale model

## Contributing 🤝
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 🙏 Acknowledgments
* MAESTRO Dataset by Magenta Team
* TensorFlow and Keras teams

---
## Model Architecture 🏗️

The system uses an encoder-decoder transformer with the following components:

### Encoder
- **Input**: Mel-spectrogram (batch, time, 80)
- **Audio Frontend**: CNN layers (128→256→d_model channels) with batch normalization
- **Position Encoding**: Learnable embeddings instead of fixed sinusoidal
- **Transformer Layers**: Multiple encoder blocks with self-attention and feed-forward networks

### Decoder  
- **Input**: Token sequences (batch, seq_len)
- **Embedding**: Token embeddings projected to d_model dimensions
- **Causal Masking**: Prevents attending to future tokens during training
- **Transformer Layers**: Decoder blocks with self-attention, cross-attention to encoder, and feed-forward networks

### Scalable Configuration
```python
# Model sizes available
SMALL  = {'d_model': 256, 'num_heads': 4, 'ff_dim': 1024, 'enc_layers': 4, 'dec_layers': 4}
MEDIUM = {'d_model': 512, 'num_heads': 8, 'ff_dim': 2048, 'enc_layers': 6, 'dec_layers': 6}
LARGE  = {'d_model': 768, 'num_heads': 12, 'ff_dim': 3072, 'enc_layers': 12, 'dec_layers': 12}
```

### Token Vocabulary
* Note Events: `P_{pitch}` for MIDI pitches (0-127)

* Time Shifts: `DT_{duration}` for relative timing

* Durations: `DU_{duration}` for note lengths

* Special Tokens: `<BOS>`, `<EOS>`, `<PAD>`

---

## Next Steps

Potential enhancements for this project:

- **User Interface**: Develop a GUI using MuseScore integration for visual score editing
- **Export Formats**: Add PDF and MusicXML (.mscz) export capabilities
- **Real-time Processing**: Implement live audio input and real-time transcription
- **Multi-instrument Support**: Extend beyond piano to recognize various instruments
- **Web Interface**: Create a browser-based version with drag-and-drop functionality