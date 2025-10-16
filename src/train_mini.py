import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter
from pathlib import Path

from src import config
from src.preprocess import audio_to_mel, midi_to_event_sequence, events_to_tokens
from src.model_full import SimpleTransformerFull  # Import the fixed version

# -----------------------------
# 1. Select a few MIDI/WAV pairs for testing
# -----------------------------
maestro_path = Path(config.MAESTRO_PATH_FULL)
test_midi_files = sorted(maestro_path.rglob("*.midi"))[:3]

# -----------------------------
# 2. Build vocabulary from these songs
# -----------------------------
all_tokens = []
for midi_file in test_midi_files:
    try:
        events = midi_to_event_sequence(midi_file)
        tokens = events_to_tokens(events)
        all_tokens.extend(tokens)
        print(f"Processed {midi_file}, got {len(tokens)} tokens")
    except Exception as e:
        print(f"Error processing {midi_file}: {e}")

# Build vocab with minimum frequency
counter = Counter(all_tokens)
vocab = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"] + [t for t, count in counter.most_common() if count >= 1]
token_to_id = {t: i for i, t in enumerate(vocab)}
id_to_token = {i: t for t, i in token_to_id.items()}
vocab_size = len(vocab)

print(f"Vocabulary size: {vocab_size}")

def tokens_to_ids(tokens, max_len=256):
    ids = [token_to_id.get(t, token_to_id["<UNK>"]) for t in tokens[:max_len-2]]
    ids = [token_to_id["<BOS>"]] + ids + [token_to_id["<EOS>"]]
    return pad_sequences([ids], maxlen=max_len, padding="post", truncating="post", value=token_to_id["<PAD>"])[0]

# -----------------------------
# 3. Create dataset with proper padding
# -----------------------------
def generator():
    for midi_file in test_midi_files:
        try:
            wav_file = midi_file.with_suffix(".wav")
            if not wav_file.exists():
                print(f"WAV file not found: {wav_file}")
                continue
                
            mel = audio_to_mel(wav_file)
            events = midi_to_event_sequence(midi_file)
            tokens = events_to_tokens(events)
            
            if len(tokens) == 0:
                print(f"No tokens generated for {midi_file}")
                continue
                
            ids = tokens_to_ids(tokens)
            
            print(f"Mel shape: {mel.shape}, Tokens length: {len(tokens)}")
            yield mel.astype(np.float32), ids.astype(np.int32)
            
        except Exception as e:
            print(f"Error in generator for {midi_file}: {e}")

# Create dataset with proper shapes
dataset_list = list(generator())
if not dataset_list:
    print("No data generated! Check your file paths and preprocessing.")
    exit()

mel_shapes = [mel.shape[0] for mel, _ in dataset_list]
token_lengths = [len(ids) for _, ids in dataset_list]

print(f"Mel lengths: {mel_shapes}")
print(f"Token lengths: {token_lengths}")

# Use fixed sizes - make sure these match your model's max_len
max_mel_len = 2048  # This should match enc_max_len in the model
max_token_len = 256  # This should match dec_max_len in the model

def preprocess_pair(mel, tokens):
    # Pad/crop mel spectrogram
    if mel.shape[0] > max_mel_len:
        mel = mel[:max_mel_len, :]
    else:
        pad_width = [(0, max_mel_len - mel.shape[0]), (0, 0)]
        mel = np.pad(mel, pad_width, mode='constant')
    
    # Ensure tokens are correct length
    if len(tokens) > max_token_len:
        tokens = tokens[:max_token_len]
    
    return mel, tokens

# Apply preprocessing
processed_data = [preprocess_pair(mel, tokens) for mel, tokens in dataset_list]

# Create TensorFlow dataset
mel_batch = np.stack([mel for mel, _ in processed_data])
token_batch = np.stack([tokens for _, tokens in processed_data])

dataset = tf.data.Dataset.from_tensor_slices((mel_batch, token_batch))
dataset = dataset.batch(2)

print(f"Final dataset - Mel batch shape: {mel_batch.shape}, Token batch shape: {token_batch.shape}")

# -----------------------------
# 4. Build model with proper max lengths
# -----------------------------
model = SimpleTransformerFull(
    vocab_size=vocab_size,
    d_model=64,
    num_heads=2,
    ff_dim=128,
    enc_layers=2,
    dec_layers=2,
    enc_max_len=max_mel_len,  # This must match your mel spectrogram length
    dec_max_len=max_token_len  # This must match your token sequence length
)

# Test forward pass
print("\nTesting forward pass...")
try:
    test_mel, test_tokens = next(iter(dataset))
    test_dec_input = test_tokens[:, :-1]
    test_logits = model((test_mel, test_dec_input), training=False)
    print(f"Forward pass successful! Logits shape: {test_logits.shape}")
except Exception as e:
    print(f"Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    exit()

# -----------------------------
# 5. Training loop
# -----------------------------
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

@tf.function
def train_step(mel_batch, token_batch):
    dec_input = token_batch[:, :-1]
    target = token_batch[:, 1:]
    
    with tf.GradientTape() as tape:
        logits = model((mel_batch, dec_input), training=True)
        loss = loss_fn(target, logits)
    
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

print("\nStarting training...")
for epoch in range(3):
    total_loss = 0
    num_batches = 0
    
    for step, (mel_batch, token_batch) in enumerate(dataset):
        loss = train_step(mel_batch, token_batch)
        total_loss += loss
        num_batches += 1
        print(f"Epoch {epoch+1}, Batch {step+1}, Loss: {loss.numpy():.4f}")
    
    if num_batches > 0:
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

# Save model
try:
    model.save("test_model_full.keras")
    print("Model saved successfully!")
except Exception as e:
    print(f"Failed to save model: {e}")

# -----------------------------
# 8. Save model, vocabulary, and config properly
# -----------------------------
import pickle
import json

def save_training_artifacts(model, token_to_id, id_to_token, model_config, file_prefix="test_model"):
    """Save all necessary files for later loading"""
    
    # Save model weights
    model.save_weights(f"{file_prefix}_weights.weights.h5")
    print(f"Model weights saved as {file_prefix}_weights.weights.h5")
    
    # Save vocabulary
    vocab_data = {
        'token_to_id': token_to_id,
        'id_to_token': id_to_token
    }
    with open(f'{file_prefix}_vocab.pkl', 'wb') as f:
        pickle.dump(vocab_data, f)
    print(f"Vocabulary saved as {file_prefix}_vocab.pkl")
    
    # Save model config
    with open(f'{file_prefix}_config.pkl', 'wb') as f:
        pickle.dump(model_config, f)
    print(f"Model config saved as {file_prefix}_config.pkl")
    
    # Also save as JSON for readability
    readable_config = {
        'vocab_size': model_config['vocab_size'],
        'd_model': model_config['d_model'],
        'num_heads': model_config['num_heads'],
        'ff_dim': model_config['ff_dim'],
        'enc_layers': model_config['enc_layers'],
        'dec_layers': model_config['dec_layers'],
        'enc_max_len': model_config['enc_max_len'],
        'dec_max_len': model_config['dec_max_len'],
        'vocab_sample': list(token_to_id.keys())[:10]  # First 10 vocab items
    }
    
    with open(f'{file_prefix}_config.json', 'w') as f:
        json.dump(readable_config, f, indent=2)
    print(f"Readable config saved as {file_prefix}_config.json")
    
    print(f"\nAll artifacts saved with prefix: {file_prefix}")

# Save everything
model_config = {
    'vocab_size': vocab_size,
    'd_model': 64,
    'num_heads': 2,
    'ff_dim': 128,
    'enc_layers': 2,
    'dec_layers': 2,
    'enc_max_len': max_mel_len,
    'dec_max_len': max_token_len
}

save_training_artifacts(model, token_to_id, id_to_token, model_config)

print("\nTraining completed successfully!")
print("Files created:")
print("- test_model_weights.weights.h5")
print("- test_model_vocab.pkl")
print("- test_model_config.pkl")
print("- test_model_config.json")