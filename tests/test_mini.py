import numpy as np
import tensorflow as tf
import pickle
from pathlib import Path

from src import config
from src.preprocess import audio_to_mel
from src.model_full import SimpleTransformerFull

def load_model_and_vocab(file_prefix="test_model"):
    """Load the trained model and vocabulary"""
    try:
        # Load vocabulary
        with open(f'{file_prefix}_vocab.pkl', 'rb') as f:
            vocab_data = pickle.load(f)
            token_to_id = vocab_data['token_to_id']
            id_to_token = vocab_data['id_to_token']
        
        # Load model config
        with open(f'{file_prefix}_config.pkl', 'rb') as f:
            model_config = pickle.load(f)
        
        print(f"Model config: {model_config}")
        
        # Recreate model architecture
        model = SimpleTransformerFull(**model_config)
        
        # BUILD the model first by calling it with dummy data
        batch_size = 1
        mel_length = model_config['enc_max_len']
        n_mels = 80
        seq_length = 10
        
        # Create dummy inputs to build the model
        dummy_mel = np.random.random((batch_size, mel_length, n_mels)).astype(np.float32)
        bos_id = token_to_id["<BOS>"]
        dummy_tokens = np.array([[bos_id] * seq_length], dtype=np.int32)
        
        # Build the model by calling it
        _ = model((dummy_mel, dummy_tokens), training=False)
        print("✅ Model built successfully")
        
        # Now load weights
        model.load_weights(f"{file_prefix}_weights.weights.h5")
        print("✅ Weights loaded successfully")
        
        print("Model and vocabulary loaded successfully!")
        print(f"Vocabulary size: {len(token_to_id)}")
        
        return model, token_to_id, id_to_token
        
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def quick_test():
    """Quick test to verify the model works"""
    print("Loading model for quick test...")
    
    model, token_to_id, id_to_token = load_model_and_vocab()
    
    if model is None:
        print("❌ Failed to load model")
        return False
    
    print("✅ Model loaded successfully!")
    
    # Test with a simple forward pass
    batch_size = 1
    mel_length = 2048  # Should match enc_max_len
    n_mels = 80
    seq_length = 10
    
    # Create dummy mel spectrogram
    dummy_mel = np.random.random((batch_size, mel_length, n_mels)).astype(np.float32)
    
    # Create decoder input starting with BOS token
    bos_id = token_to_id["<BOS>"]
    dummy_tokens = np.array([[bos_id] * seq_length], dtype=np.int32)
    
    try:
        # Test forward pass
        logits = model((dummy_mel, dummy_tokens), training=False)
        print(f"✅ Forward pass successful!")
        print(f"   Input shapes: mel{dummy_mel.shape}, tokens{dummy_tokens.shape}")
        print(f"   Output shape: {logits.shape}")
        
        # Show some predictions
        predicted_ids = tf.argmax(logits, axis=-1).numpy()[0]
        predicted_tokens = [id_to_token.get(i, "<UNK>") for i in predicted_ids[:5]]
        print(f"   First 5 predicted tokens: {predicted_tokens}")
        
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_generation():
    """Test sequence generation"""
    print("\nTesting sequence generation...")
    
    model, token_to_id, id_to_token = load_model_and_vocab()
    
    if model is None:
        return
    
    try:
        # Create dummy mel spectrogram
        mel_length = model.enc_max_len
        dummy_mel = np.random.random((1, mel_length, 80)).astype(np.float32)
        
        # Start with just BOS token
        bos_id = token_to_id["<BOS>"]
        eos_id = token_to_id["<EOS>"]
        
        generated_tokens = [bos_id]
        max_length = 20  # Short for testing
        
        print("Generating sequence...")
        for step in range(max_length):
            # Prepare inputs
            dec_input = np.array([generated_tokens], dtype=np.int32)
            
            # Get predictions
            logits = model((dummy_mel, dec_input), training=False)
            
            # Get next token (greedy)
            next_token_logits = logits[0, -1, :]
            next_token = tf.argmax(next_token_logits).numpy()
            
            generated_tokens.append(next_token)
            
            # Stop if EOS
            if next_token == eos_id:
                print(f"EOS token generated at step {step}")
                break
        
        # Convert to tokens
        token_sequence = [id_to_token.get(tok_id, "<UNK>") for tok_id in generated_tokens[1:]]  # Skip BOS
        print(f"Generated sequence ({len(token_sequence)} tokens): {token_sequence}")
        
        return True
        
    except Exception as e:
        print(f"❌ Generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_files_exist():
    """Check if all required files exist"""
    print("Checking for required files...")
    
    required_files = [
        'test_model_weights.weights.h5',
        'test_model_vocab.pkl', 
        'test_model_config.pkl'
    ]
    
    all_exist = True
    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file} - Found")
        else:
            print(f"❌ {file} - Missing")
            all_exist = False
    
    return all_exist

if __name__ == "__main__":
    print("Running model tests...")
    
    # First check if files exist
    if not check_files_exist():
        print("\n❌ Missing required files! Please run the training script first.")
        print("Run: python train_mini.py")
        exit(1)
    
    print("\n" + "="*50)
    
    # Run tests
    success = quick_test()
    
    if success:
        print("\n" + "="*50)
        test_generation()
    
    print("\nTest completed!")