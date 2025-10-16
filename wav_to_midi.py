import numpy as np
import tensorflow as tf
import pickle
from pathlib import Path
import argparse
from src import config
from src.preprocess import audio_to_mel, tokens_to_events, events_to_midi
from src.model_full import SimpleTransformerFull

class WavToMidiConverter:
    def __init__(self, model_prefix="test_model"):
        """Initialize the converter with trained model"""
        self.model_prefix = model_prefix
        self.model = None
        self.token_to_id = None
        self.id_to_token = None
        
        self.load_model()
    
    def load_model(self):
        """Load the trained model and vocabulary"""
        try:
            # Load vocabulary
            with open(f'{self.model_prefix}_vocab.pkl', 'rb') as f:
                vocab_data = pickle.load(f)
                self.token_to_id = vocab_data['token_to_id']
                self.id_to_token = vocab_data['id_to_token']
            
            # Load model config
            with open(f'{self.model_prefix}_config.pkl', 'rb') as f:
                model_config = pickle.load(f)
            
            # Create and build model
            self.model = SimpleTransformerFull(**model_config)
            
            # Build model with dummy data
            dummy_mel = np.random.random((1, model_config['enc_max_len'], 80)).astype(np.float32)
            dummy_tokens = np.array([[self.token_to_id["<BOS>"]] * 10], dtype=np.int32)
            _ = self.model((dummy_mel, dummy_tokens), training=False)
            
            # Load weights
            self.model.load_weights(f"{self.model_prefix}_weights.weights.h5")
            
            print(f"✅ Model loaded successfully! Vocabulary size: {len(self.token_to_id)}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def generate_sequence(self, mel_input, max_length=500, temperature=0.8):
        """Generate token sequence from mel spectrogram"""
        # Prepare mel input
        if mel_input.shape[0] > self.model.enc_max_len:
            mel_input = mel_input[:self.model.enc_max_len, :]
        else:
            pad_width = [(0, self.model.enc_max_len - mel_input.shape[0]), (0, 0)]
            mel_input = np.pad(mel_input, pad_width, mode='constant')
        
        mel_batch = mel_input[np.newaxis, ...]
        
        # Start with BOS token
        bos_id = self.token_to_id["<BOS>"]
        eos_id = self.token_to_id["<EOS>"]
        
        generated_tokens = [bos_id]
        
        print("Generating MIDI sequence...")
        for step in range(max_length):
            # Prepare decoder input
            dec_input = np.array([generated_tokens], dtype=np.int32)
            
            # Get model predictions
            logits = self.model((mel_batch, dec_input), training=False)
            
            # Get next token probabilities
            next_token_logits = logits[0, -1, :] / temperature
            probabilities = tf.nn.softmax(next_token_logits).numpy()
            
            # Sample from distribution (you can use greedy by setting temperature very low)
            next_token = np.random.choice(len(probabilities), p=probabilities)
            
            # Stop if EOS token or if we have a reasonable sequence
            if next_token == eos_id:
                print(f"🎵 EOS token generated at step {step}")
                break
                
            generated_tokens.append(next_token)
            
            # Progress indicator
            if step % 100 == 0:
                current_tokens = [self.id_to_token.get(t, "<UNK>") for t in generated_tokens[1:]]
                note_count = sum(1 for t in current_tokens if t.startswith('P_'))
                print(f"Step {step}: Generated {len(current_tokens)} tokens, {note_count} notes")
        
        # Convert to token strings
        token_sequence = [self.id_to_token.get(tok_id, "<UNK>") for tok_id in generated_tokens[1:]]
        
        # Filter out unknown tokens
        token_sequence = [t for t in token_sequence if t != "<UNK>"]
        
        print(f"🎵 Generated {len(token_sequence)} tokens total")
        print(f"🎵 Note events: {sum(1 for t in token_sequence if t.startswith('P_'))}")
        print(f"🎵 Time shifts: {sum(1 for t in token_sequence if t.startswith('DT_'))}")
        print(f"🎵 Durations: {sum(1 for t in token_sequence if t.startswith('DU_'))}")
        
        return token_sequence
    
    def convert_wav_to_midi(self, wav_path, output_midi_path=None, max_length=1000):
        """Convert WAV file to MIDI file"""
        if output_midi_path is None:
            output_midi_path = Path(wav_path).with_suffix('.mid')
        
        print(f"🎵 Converting {wav_path} to MIDI...")
        
        # Process audio to mel spectrogram
        mel = audio_to_mel(wav_path)
        print(f"🎵 Mel spectrogram shape: {mel.shape}")
        
        # Generate token sequence
        tokens = self.generate_sequence(mel, max_length=max_length)
        
        if not tokens:
            print("❌ No tokens generated!")
            return None
        
        # Convert tokens to events
        events = tokens_to_events(tokens)
        print(f"🎵 Converted to {len(events)} events")
        
        # Convert events to MIDI
        midi_stream = events_to_midi(events, output_midi_path)
        
        print(f"✅ Successfully created: {output_midi_path}")
        return output_midi_path

def main():
    parser = argparse.ArgumentParser(description='Convert WAV file to MIDI')
    parser.add_argument('wav_file', help='Input WAV file path')
    parser.add_argument('--output', '-o', help='Output MIDI file path (optional)')
    parser.add_argument('--max_length', '-l', type=int, default=1000, 
                       help='Maximum sequence length (default: 1000)')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.wav_file).exists():
        print(f"❌ File not found: {args.wav_file}")
        return
    
    # Initialize converter
    converter = WavToMidiConverter()
    
    # Convert!
    converter.convert_wav_to_midi(
        wav_path=args.wav_file,
        output_midi_path=args.output,
        max_length=args.max_length
    )

if __name__ == "__main__":
    main()