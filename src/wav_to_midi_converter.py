import tensorflow as tf
import numpy as np
import pickle
import argparse
from pathlib import Path
import sys
import os
import re

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_scalable import ScalableTransformer
from src.preprocess import audio_to_mel, tokens_to_events, events_to_midi

class WavToMidiConverter:
    def __init__(self, model_weights_path, vocab_path):
        """Initialize the converter with trained model"""
        self.model_weights_path = model_weights_path
        self.vocab_path = vocab_path
        self.model = None
        self.vocab = None
        
        self.load_model_and_vocab()
    
    def detect_model_size(self):
        """Detect model size from weights filename"""
        filename = Path(self.model_weights_path).name.lower()
        
        if 'small' in filename:
            return 'small'
        elif 'medium' in filename:
            return 'medium'
        elif 'large' in filename:
            return 'large'
        else:
            # Try to extract from pattern like scalable_model_medium.weights.h5
            match = re.search(r'scalable_model_(\w+)\.weights', filename)
            if match:
                return match.group(1)
            else:
                print("⚠️  Could not detect model size from filename, defaulting to 'small'")
                return 'small'
    
    def get_model_config(self, model_size):
        """Get model configuration for different sizes"""
        MODEL_CONFIGS = {
            'small': {
                'd_model': 256,
                'num_heads': 4,
                'ff_dim': 1024,
                'enc_layers': 4,
                'dec_layers': 4
            },
            'medium': {
                'd_model': 512,
                'num_heads': 8,
                'ff_dim': 2048,
                'enc_layers': 6,
                'dec_layers': 6
            },
            'large': {
                'd_model': 768,
                'num_heads': 12,
                'ff_dim': 3072,
                'enc_layers': 12,
                'dec_layers': 12
            }
        }
        
        if model_size not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model size: {model_size}. Available: {list(MODEL_CONFIGS.keys())}")
        
        return MODEL_CONFIGS[model_size]
    
    def load_model_and_vocab(self):
        """Load the trained model and vocabulary"""
        print("Loading model and vocabulary...")
        
        # Load vocabulary
        with open(self.vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)
        
        vocab_size = len(self.vocab['token_to_id'])
        print(f"✅ Vocabulary loaded: {vocab_size} tokens")
        
        # Detect model size from filename
        model_size = self.detect_model_size()
        print(f"🎯 Detected model size: {model_size}")
        
        # Get model configuration
        model_config = self.get_model_config(model_size)
        model_config['vocab_size'] = vocab_size
        
        print(f"📊 Model architecture:")
        for key, value in model_config.items():
            if key != 'vocab_size':
                print(f"   {key}: {value}")
        
        # Create model
        self.model = ScalableTransformer(**model_config)
        
        # Build model with dummy data
        dummy_mel = tf.random.normal((1, 2048, 80))
        dummy_tokens = tf.ones((1, 100), dtype=tf.int32)
        _ = self.model([dummy_mel, dummy_tokens])
        
        # Load weights
        self.model.load_weights(self.model_weights_path)
        print("✅ Model weights loaded successfully")
    
    def prepare_mel_spectrogram(self, mel_spectrogram, target_length=2048):
        """Prepare mel spectrogram for model input"""
        if mel_spectrogram.shape[0] > target_length:
            # Truncate if too long
            mel_spectrogram = mel_spectrogram[:target_length, :]
            print(f"📏 Truncated mel spectrogram to {target_length} frames")
        else:
            # Pad if too short
            pad_width = [(0, target_length - mel_spectrogram.shape[0]), (0, 0)]
            mel_spectrogram = np.pad(mel_spectrogram, pad_width, mode='constant')
            print(f"📏 Padded mel spectrogram to {target_length} frames")
        
        return mel_spectrogram[np.newaxis, ...]  # Add batch dimension
    
    def generate_sequence(self, mel_spectrogram, max_length=1000, temperature=0.8, top_k=50):
        """Generate MIDI sequence from mel spectrogram"""
        # Prepare mel input
        mel_batch = self.prepare_mel_spectrogram(mel_spectrogram)
        
        # Get special tokens
        bos_id = self.vocab['token_to_id']['<BOS>']
        eos_id = self.vocab['token_to_id']['<EOS>']
        pad_id = self.vocab['token_to_id']['<PAD>']
        
        generated_ids = [bos_id]
        
        print(f"🎵 Generating sequence (max_length={max_length}, temperature={temperature})...")
        
        for step in range(max_length):
            # Prepare decoder input
            dec_input = np.array([generated_ids], dtype=np.int32)
            
            # Get model predictions
            logits = self.model([mel_batch, dec_input], training=False)
            
            # Get next token probabilities
            next_token_logits = logits[0, -1, :] / temperature
            
            # Apply top-k filtering if specified
            if top_k > 0:
                indices_to_remove = next_token_logits < tf.math.top_k(next_token_logits, top_k).values[-1]
                next_token_logits = tf.where(indices_to_remove, -float('inf'), next_token_logits)
            
            probabilities = tf.nn.softmax(next_token_logits).numpy()
            
            # Sample from distribution
            next_token = np.random.choice(len(probabilities), p=probabilities)
            
            # Stop if EOS token
            if next_token == eos_id:
                print(f"🛑 EOS token generated at step {step}")
                break
                
            # Don't add PAD tokens during generation
            if next_token != pad_id:
                generated_ids.append(next_token)
            
            # Progress indicator
            if step % 100 == 0:
                current_tokens = [self.vocab['id_to_token'].get(t, '<UNK>') for t in generated_ids[1:]]
                note_count = sum(1 for t in current_tokens if t.startswith('P_'))
                print(f"  Step {step}: {len(current_tokens)} tokens, {note_count} notes")
        
        # Convert to token strings (skip BOS token)
        token_sequence = [self.vocab['id_to_token'].get(tok_id, '<UNK>') 
                         for tok_id in generated_ids[1:]]
        
        # Filter out unknown and padding tokens
        token_sequence = [t for t in token_sequence if t not in ['<UNK>', '<PAD>']]
        
        print(f"🎵 Generation completed:")
        print(f"   Total tokens: {len(token_sequence)}")
        print(f"   Note events: {sum(1 for t in token_sequence if t.startswith('P_'))}")
        print(f"   Time shifts: {sum(1 for t in token_sequence if t.startswith('DT_'))}")
        print(f"   Durations: {sum(1 for t in token_sequence if t.startswith('DU_'))}")
        
        return token_sequence
    
    def convert_wav_to_midi(self, wav_path, output_path=None, max_length=1000, temperature=0.8):
        """Convert WAV file to MIDI file"""
        if output_path is None:
            wav_name = Path(wav_path).stem
            output_path = f"{wav_name}_converted.mid"
        
        print(f"🎵 Converting: {Path(wav_path).name}")
        print(f"🎯 Output: {output_path}")
        
        # Process audio to mel spectrogram
        print("🔊 Processing audio...")
        mel = audio_to_mel(wav_path)
        if mel is None:
            print("❌ Failed to process audio file")
            return None
        
        print(f"✅ Mel spectrogram shape: {mel.shape}")
        
        # Generate token sequence
        tokens = self.generate_sequence(mel, max_length=max_length, temperature=temperature)
        
        if not tokens:
            print("❌ No tokens generated!")
            return None
        
        print(f"🔍 Sample tokens: {tokens[:15]}")
        
        # Convert tokens to events
        print("🔄 Converting tokens to events...")
        events = tokens_to_events(tokens)
        print(f"✅ Converted to {len(events)} events")
        
        # Convert events to MIDI
        print("🎹 Creating MIDI file...")
        midi_stream = events_to_midi(events, output_path)
        
        if midi_stream:
            print(f"✅ Successfully created: {output_path}")
            return output_path
        else:
            print("❌ Failed to create MIDI file")
            return None

def main():
    parser = argparse.ArgumentParser(
        description='Convert WAV to MIDI using trained transformer model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python wav_to_midi.py input.wav
  python wav_to_midi.py input.wav --output my_song.mid
  python wav_to_midi.py input.wav --model-weights scalable_model_medium.weights.h5
  python wav_to_midi.py input.wav --max-length 2000 --temperature 0.5
        '''
    )
    
    parser = argparse.ArgumentParser(description='Convert WAV to MIDI using trained model')
    parser.add_argument('wav_file', help='Input WAV file path')
    parser.add_argument('--output', '-o', help='Output MIDI file path (optional)')
    parser.add_argument('--max-length', '-l', type=int, default=1000, 
                       help='Maximum sequence length (default: 1000)')
    parser.add_argument('--temperature', '-t', type=float, default=0.8,
                       help='Sampling temperature (lower = more deterministic, default: 0.8)')
    parser.add_argument('--model-weights', '-m', 
                       default='scalable_model_small.weights.h5',
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
        print("💡 Make sure you've trained the model first!")
        print("💡 Available model files:")
        for f in Path('.').glob('scalable_model_*.weights.h5'):
            print(f"   - {f.name}")
        return
    
    if not Path(args.vocab).exists():
        print(f"❌ Vocabulary file not found: {args.vocab}")
        print("💡 Make sure you have the vocabulary file from preprocessing")
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
        max_length=args.max_length,
        temperature=args.temperature
    )
    
    if result:
        print(f"\n🎉 Conversion successful!")
        print(f"📁 MIDI file: {result}")
        print("🎹 You can open this file in any DAW or music notation software!")
    else:
        print("\n❌ Conversion failed!")

if __name__ == "__main__":
    main()