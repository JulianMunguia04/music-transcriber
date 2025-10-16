import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter

from src import config
from src.preprocess import audio_to_mel, midi_to_events, events_to_tokens
from src.model_full import SimpleTransformerFull  # your bigger model

# -----------------------------
# 1. Helper functions
# -----------------------------
def build_vocab(token_lists):
    """Build vocabulary from all token lists."""
    counter = Counter(token for tokens in token_lists for token in tokens)
    vocab = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"] + [t for t, _ in counter.most_common()]
    token_to_id = {t: i for i, t in enumerate(vocab)}
    id_to_token = {i: t for t, i in token_to_id.items()}
    return vocab, token_to_id, id_to_token

def tokens_to_ids(tokens, token_to_id, max_len=config.MAX_LEN):
    """Convert token list to fixed-length ID sequence."""
    ids = [token_to_id.get(t, token_to_id["<UNK>"]) for t in tokens]
    ids = [token_to_id["<BOS>"]] + ids + [token_to_id["<EOS>"]]
    return pad_sequences([ids], maxlen=max_len, padding="post", truncating="post")[0]

def dataset_generator(maestro_path, token_to_id):
    """Generator that yields (mel, token_ids) for each song."""
    maestro_path = Path(maestro_path)
    midi_files = sorted(maestro_path.rglob("*.midi"))
    for midi_file in midi_files:
        wav_file = midi_file.with_suffix(".wav")
        if not wav_file.exists():
            continue
        mel = audio_to_mel(wav_file)
        events = midi_to_events(midi_file)
        tokens = events_to_tokens(events)
        ids = tokens_to_ids(tokens, token_to_id)
        yield mel.astype(np.float32), ids.astype(np.int32)

# -----------------------------
# 2. Prepare dataset
# -----------------------------
# First, build vocabulary from a small sample
sample_midi_files = list(Path(config.MAESTRO_PATH).rglob("*.midi"))[:50]
sample_tokens = []
for midi_file in sample_midi_files:
    events = midi_to_events(midi_file)
    sample_tokens.append(events_to_tokens(events))

vocab, token_to_id, id_to_token = build_vocab(sample_tokens)
vocab_size = len(vocab)

# Dataset
ds = tf.data.Dataset.from_generator(
    lambda: dataset_generator(config.MAESTRO_PATH, token_to_id),
    output_signature=(
        tf.TensorSpec(shape=(None, config.N_MELS), dtype=tf.float32),
        tf.TensorSpec(shape=(config.MAX_LEN,), dtype=tf.int32)
    )
)
ds = ds.padded_batch(config.BATCH_SIZE, padded_shapes=([None, config.N_MELS], [config.MAX_LEN]))
ds = ds.prefetch(tf.data.AUTOTUNE)

# -----------------------------
# 3. Build model
# -----------------------------
model = SimpleTransformerFull(
    vocab_size=vocab_size,
    d_model=256,      # larger model
    num_heads=4,
    ff_dim=512,
    enc_layers=4,
    dec_layers=4,
    max_len=config.MAX_LEN
)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# -----------------------------
# 4. Checkpointing
# -----------------------------
checkpoint_dir = Path(config.OUTPUT_PATH) / "checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)
ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=5)

# -----------------------------
# 5. Training loop
# -----------------------------
EPOCHS = 10  # adjust as needed

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    for step, (mel_batch, dec_batch) in enumerate(ds):
        dec_input = dec_batch[:, :-1]
        target = dec_batch[:, 1:]

        with tf.GradientTape() as tape:
            logits = model((mel_batch, dec_input), training=True)
            loss = loss_fn(target, logits)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss.numpy():.4f}")

    ckpt_manager.save()
    print(f"Checkpoint saved at epoch {epoch+1}")
