# model_full_fixed.py
import tensorflow as tf
from tensorflow.keras import layers

# -----------------------------
# Fixed Positional Encoding Layer
# -----------------------------
class PositionalEmbedding(layers.Layer):
    """
    Adds a learnable positional embedding to token indices.
    """
    def __init__(self, max_len, d_model):
        super().__init__()
        self.max_len = max_len
        self.pos_emb = layers.Embedding(input_dim=max_len, output_dim=d_model)

    def call(self, x):
        # x shape: (batch_size, seq_len, d_model)
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        # Ensure we don't exceed max_len
        positions = tf.range(seq_len)  # (seq_len,)
        positions = tf.minimum(positions, self.max_len - 1)  # Cap positions to max_len-1
        positions = tf.expand_dims(positions, 0)  # (1, seq_len)
        positions = tf.tile(positions, [batch_size, 1])  # (batch_size, seq_len)
        pos_embeddings = self.pos_emb(positions)  # (batch_size, seq_len, d_model)
        return pos_embeddings

# -----------------------------
# Transformer Encoder Block (unchanged)
# -----------------------------
def transformer_encoder_block(d_model, num_heads, ff_dim, dropout=0.1):
    inputs = layers.Input(shape=(None, d_model))
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(inputs, inputs)
    attn = layers.Dropout(dropout)(attn)
    out1 = layers.LayerNormalization(epsilon=1e-6)(inputs + attn)

    ffn = layers.Dense(ff_dim, activation='relu')(out1)
    ffn = layers.Dense(d_model)(ffn)
    ffn = layers.Dropout(dropout)(ffn)
    out2 = layers.LayerNormalization(epsilon=1e-6)(out1 + ffn)
    return tf.keras.Model(inputs=inputs, outputs=out2, name="encoder_block")

# -----------------------------
# Transformer Decoder Block (unchanged)
# -----------------------------
def transformer_decoder_block(d_model, num_heads, ff_dim, dropout=0.1):
    enc_inputs = layers.Input(shape=(None, d_model))
    dec_inputs = layers.Input(shape=(None, d_model))

    attn1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(dec_inputs, dec_inputs)
    attn1 = layers.Dropout(dropout)(attn1)
    out1 = layers.LayerNormalization(epsilon=1e-6)(dec_inputs + attn1)

    attn2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(out1, enc_inputs)
    attn2 = layers.Dropout(dropout)(attn2)
    out2 = layers.LayerNormalization(epsilon=1e-6)(out1 + attn2)

    ffn = layers.Dense(ff_dim, activation='relu')(out2)
    ffn = layers.Dense(d_model)(ffn)
    ffn = layers.Dropout(dropout)(ffn)
    out3 = layers.LayerNormalization(epsilon=1e-6)(out2 + ffn)
    return tf.keras.Model(inputs=[enc_inputs, dec_inputs], outputs=out3, name="decoder_block")

# -----------------------------
# Full Transformer Model with proper max_len handling
# -----------------------------
class SimpleTransformerFull(tf.keras.Model):
    def __init__(self, vocab_size, d_model=256, num_heads=4, ff_dim=512,
                 enc_layers=4, dec_layers=4, max_len=8192, enc_max_len=8192, dec_max_len=8192):
        super().__init__()
        self.d_model = d_model
        
        # Separate max lengths for encoder and decoder
        self.enc_max_len = enc_max_len
        self.dec_max_len = dec_max_len

        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=d_model)
        self.pos_emb_enc = PositionalEmbedding(max_len=enc_max_len, d_model=d_model)
        self.pos_emb_dec = PositionalEmbedding(max_len=dec_max_len, d_model=d_model)

        self.enc_layers = [transformer_encoder_block(d_model, num_heads, ff_dim) for _ in range(enc_layers)]
        self.dec_layers = [transformer_decoder_block(d_model, num_heads, ff_dim) for _ in range(dec_layers)]
        self.out = layers.Dense(vocab_size)

        # Use a Dense projection for encoder inputs
        self.enc_proj = layers.Dense(d_model)

    def call(self, inputs, training=False):
        enc_inputs, dec_tokens = inputs  # enc_inputs: (batch, time, features)

        # Project encoder inputs to d_model
        x = self.enc_proj(enc_inputs)
        x += self.pos_emb_enc(x)

        # Pass through encoder layers
        for layer in self.enc_layers:
            x = layer(x)

        # Decoder embeddings
        y = self.token_emb(dec_tokens)
        y += self.pos_emb_dec(y)

        # Pass through decoder layers
        for layer in self.dec_layers:
            y = layer([x, y])

        # Project to vocabulary
        logits = self.out(y)
        return logits

    def get_config(self):
        return {
            "d_model": self.d_model,
            "enc_max_len": self.enc_max_len,
            "dec_max_len": self.dec_max_len
        }