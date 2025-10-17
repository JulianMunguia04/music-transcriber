import tensorflow as tf
import numpy as np
import pickle
import argparse
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_scalable import ScalableTransformer
from src.preprocess import audio_to_mel, tokens_to_events, events_to_midi
from src.config import MAESTRO_PATH_FULL

class WavToMidiConverter:
    def __init__(self, model_weights_path, vocab_path):
        """Initialize the converter with trained model"""
        self.model_weights_path = model_weights_path
        self.vocab_path = vocab_path
        self.model = None
        self.vocab = None
        
        self.load_model_and_vocab()
    
    def load_model_and_vocab(self):
        """Load the trained model and vocabulary"""
        print("Loading model and vocabulary...")
        
        # Load vocabulary
        with open(self.vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)
        
        vocab_size = len(self.vocab['token_to_id'])
        print(f"✅ Vocabulary loaded: {vocab_size} tokens")
        
        # Create model with same architecture as training
        model_config = {
            'vocab_size': vocab_size,
            'd_model': 256,      # Must match your training!
            'num_heads': 4,      # Must match your training!
            'ff_dim': 1024,      # Must match your training!
            'enc_layers': 4,     # Must match your training!
            'dec_layers': 4,     # Must match your training!
        }
        
        self.model = ScalableTransformer(**model_config)
        
        # Build model with dummy data
        dummy_mel = tf.random.normal((1, 2048, 80))
        dummy_tokens = tf.ones((1, 100), dtype=tf.int32)
        _ = self.model([dummy_mel, dummy_tokens])
        
        # Load weights
        self.model.load_weights(self.model_weights_path)
        print("✅ Model weights loaded")
    
    def generate_sequence(self, mel_spectrogram, max_length=500, temperature=0.8):
        """Generate MIDI sequence from mel spectrogram"""
        # Prepare mel input
        if mel_spectrogram.shape[0] > 2048:
            mel_spectrogram = mel_spectrogram[:2048, :]
        else:
            pad_width = [(0, 2048 - mel_spectrogram.shape[0]), (0, 0)]
            mel_spectrogram = np.pad(mel_spectrogram, pad_width, mode='constant')
        
        mel_batch = mel_spectrogram[np.newaxis, ...]  # Add batch dimension
        
        # Start with BOS token
        bos_id = self.vocab['token_to_id']['<BOS>']
        eos_id = self.vocab['token_to_id']['<EOS>']
        pad_id = self.vocab['token_to_id']['<PAD>']
        
        generated_ids = [bos_id]
        
        print("Generating MIDI sequence...")
        for step in range(max_length):
            # Prepare decoder input
            dec_input = np.array([generated_ids], dtype=np.int32)
            
            # Get model predictions
            logits = self.model([mel_batch, dec_input], training=False)
            
            # Get next token probabilities
            next_token_logits = logits[0, -1, :] / temperature
            probabilities = tf.nn.softmax(next_token_logits).numpy()
            
            # Sample from distribution (you can use greedy by setting temperature very low)
            next_token = np.random.choice(len(probabilities), p=probabilities)
            
            # Stop if EOS token
            if next_token == eos_id:
                print(f"🎵 EOS token generated at step {step}")
                break
                
            # Don't add PAD tokens during generation
            if next_token != pad_id:
                generated_ids.append(next_token)
            
            # Progress indicator
            if step % 50 == 0:
                current_tokens = [self.vocab['id_to_token'].get(t, '<UNK>') for t in generated_ids[1:]]
                note_count = sum(1 for t in current_tokens if t.startswith('P_'))
                print(f"  Step {step}: {len(current_tokens)} tokens, {note_count} notes")
        
        # Convert to token strings (skip BOS token)
        token_sequence = [self.vocab['id_to_token'].get(tok_id, '<UNK>') 
                         for tok_id in generated_ids[1:]]
        
        # Filter out unknown and padding tokens
        token_sequence = [t for t in token_sequence if t not in ['<UNK>', '<PAD>']]
        
        print(f"🎵 Generated {len(token_sequence)} tokens")
        print(f"🎵 Note events: {sum(1 for t in token_sequence if t.startswith('P_'))}")
        print(f"🎵 Time shifts: {sum(1 for t in token_sequence if t.startswith('DT_'))}")
        print(f"🎵 Durations: {sum(1 for t in token_sequence if t.startswith('DU_'))}")
        
        return token_sequence
    
    def convert_wav_to_midi(self, wav_path, output_path=None, max_length=500):
        """Convert WAV file to MIDI file"""
        if output_path is None:
            wav_name = Path(wav_path).stem
            output_path = f"{wav_name}_converted.mid"
        
        print(f"🎵 Converting: {Path(wav_path).name}")
        print(f"🎯 Output: {output_path}")
        
        # Process audio to mel spectrogram
        print("Processing audio...")
        mel = audio_to_mel(wav_path)
        if mel is None:
            print("❌ Failed to process audio file")
            return None
        
        print(f"✅ Mel spectrogram: {mel.shape}")
        
        # Generate token sequence
        tokens = self.generate_sequence(mel, max_length=max_length)
        
        if not tokens:
            print("❌ No tokens generated!")
            return None
        
        print(f"Sample tokens: {tokens[:10]}")
        
        # Convert tokens to events
        events = tokens_to_events(tokens)
        print(f"✅ Converted to {len(events)} events")
        
        # Convert events to MIDI
        print("Creating MIDI file...")
        midi_stream = events_to_midi(events, output_path)
        
        if midi_stream:
            print(f"✅ Successfully created: {output_path}")
            return output_path
        else:
            print("❌ Failed to create MIDI file")
            return None

def main():
    parser = argparse.ArgumentParser(description='Convert WAV to MIDI using trained model')
    parser.add_argument('wav_file', help='Input WAV file path')
    parser.add_argument('--output', '-o', help='Output MIDI file path (optional)')
    parser.add_argument('--max-length', '-l', type=int, default=500, 
                       help='Maximum sequence length (default: 500)')
    parser.add_argument('--model-weights', '-m', default='scalable_model_small.weights.h5',
                       help='Model weights file (default: scalable_model_small.weights.h5)')
    parser.add_argument('--vocab', '-v', default='full_vocab.pkl',
                       help='Vocabulary file (default: full_vocab.pkl)')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not Path(args.wav_file).exists():
        print(f"❌ WAV file not found: {args.wav_file}")
        return
    
    if not Path(args.model_weights).exists():
        print(f"❌ Model weights not found: {args.model_weights}")
        print("Make sure you've trained the model first!")
        return
    
    if not Path(args.vocab).exists():
        print(f"❌ Vocabulary file not found: {args.vocab}")
        return
    
    # Initialize converter
    converter = WavToMidiConverter(
        model_weights_path=args.model_weights,
        vocab_path=args.vocab
    )
    
    # Convert!
    result = converter.convert_wav_to_midi(
        wav_path=args.wav_file,
        output_path=args.output,
        max_length=args.max_length
    )
    
    if result:
        print(f"\n🎉 Conversion successful!")
        print(f"📁 MIDI file: {result}")
        print("You can open this file in any DAW or music notation software!")
    else:
        print("\n❌ Conversion failed!")

if __name__ == "__main__":
    main()