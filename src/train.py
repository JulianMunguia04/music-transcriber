import numpy as np
import tensorflow as tf
from .preprocess import audio_to_mel, midi_to_event_sequence, events_to_tokens
from . import config
from .model import SimpleTransformer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter
from pathlib import Path

# -----------------------------
# 1. Load example WAV + MIDI
# -----------------------------
wav_path = Path(config.MAESTRO_PATH) / "2018/MUS-2018_01/song1.wav"
midi_path = Path(config.MAESTRO_PATH) / "2018/MUS-2018_01/song1.midi"

# Convert audio to mel
X_mel = audio_to_mel(wav_path)
# Truncate mel to MAX_MEL_LEN to match positional embedding
MAX_MEL_LEN = 5000
X_mel = X_mel[:MAX_MEL_LEN, :]

# Convert MIDI to events and then tokens
events = midi_to_event_sequence(midi_path)
token_list = events_to_tokens(events)

# -----------------------------
# 2. Build vocabulary
# -----------------------------
counter = Counter(token_list)
vocab = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"] + [t for t, _ in counter.most_common()]
token_to_id = {t: i for i, t in enumerate(vocab)}
id_to_token = {i: t for t, i in token_to_id.items()}
vocab_size = len(vocab)

# -----------------------------
# 3. Tokenize and pad decoder sequence
# -----------------------------
DEC_MAXLEN = 256

def toks_to_ids(toks, maxlen=DEC_MAXLEN):
    ids = [token_to_id.get(t, token_to_id["<UNK>"]) for t in toks]
    ids = [token_to_id["<BOS>"]] + ids + [token_to_id["<EOS>"]]
    ids = ids[:maxlen]  # truncate if too long
    return pad_sequences([ids], maxlen=maxlen, padding="post", truncating="post")[0]

y_ids = toks_to_ids(token_list, maxlen=DEC_MAXLEN)

# -----------------------------
# 4. Create tf.data Dataset
# -----------------------------
def gen():
    yield X_mel, y_ids

ds = tf.data.Dataset.from_generator(
    gen,
    output_signature=(
        tf.TensorSpec(shape=(None, config.N_MELS), dtype=tf.float32),
        tf.TensorSpec(shape=(DEC_MAXLEN,), dtype=tf.int32),
    )
)
ds = ds.padded_batch(1, padded_shapes=([None, config.N_MELS], [DEC_MAXLEN]))

# -----------------------------
# 5. Build model
# -----------------------------
model = SimpleTransformer(
    vocab_size=vocab_size,
    d_model=128,
    num_heads=2,
    ff_dim=256,
    enc_layers=2,
    dec_layers=2,
    max_len=MAX_MEL_LEN  # encoder positional embedding length
)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# -----------------------------
# 6. Training loop (tiny prototype)
# -----------------------------
for epoch in range(3):
    print(f"\nEpoch {epoch+1}")
    for step, (mel_batch, dec_batch) in enumerate(ds):
        dec_input = dec_batch[:, :-1]
        target = dec_batch[:, 1:]

        with tf.GradientTape() as tape:
            logits = model((mel_batch, dec_input), training=True)
            loss = loss_fn(target, logits)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        print(f"Step {step+1}, Loss: {loss.numpy():.4f}")

# -----------------------------
# 7. Generate prediction
# -----------------------------
dec_in = dec_batch[:, :-1]
pred_logits = model((mel_batch, dec_in))
pred_ids = tf.argmax(pred_logits, axis=-1).numpy()[0]
pred_tokens = [id_to_token[i] for i in pred_ids]

print("\nPredicted tokens (first 50):")
print(pred_tokens[:50])
