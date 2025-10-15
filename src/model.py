import tensorflow as tf
from tensorflow.keras import layers

class SimpleTransformer(tf.keras.Model):
    def __init__(self, vocab_size, d_model=128, num_heads=2, ff_dim=256,
                 enc_layers=2, dec_layers=2, max_len=2048):
        super().__init__()

        # -------------------
        # Encoder: mel input
        # -------------------
        self.input_conv = tf.keras.Sequential([
            layers.Conv1D(d_model, kernel_size=3, padding="same", activation="relu"),
            layers.LayerNormalization()
        ])

        self.pos_emb_enc = layers.Embedding(input_dim=max_len, output_dim=d_model)

        self.enc_blocks = []
        for _ in range(enc_layers):
            self.enc_blocks.append([
                layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads),
                layers.LayerNormalization(),
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(d_model),
                layers.LayerNormalization()
            ])

        # -------------------
        # Decoder: token input
        # -------------------
        self.token_emb = layers.Embedding(vocab_size, d_model)
        self.pos_emb_dec = layers.Embedding(input_dim=max_len, output_dim=d_model)

        self.dec_blocks = []
        for _ in range(dec_layers):
            self.dec_blocks.append([
                layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads),  # self-attention
                layers.LayerNormalization(),
                layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads),  # enc-dec attention
                layers.LayerNormalization(),
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(d_model),
                layers.LayerNormalization()
            ])

        # Output projection
        self.final_dense = layers.Dense(vocab_size)

    def call(self, inputs, training=False):
        mel, dec_in = inputs  # mel: (B, T_mel, n_mels), dec_in: (B, T_tok)

        # -------------------
        # Encoder
        # -------------------
        x = self.input_conv(mel)
        pos_idx = tf.range(tf.shape(x)[1])
        x += self.pos_emb_enc(pos_idx)

        for attn, norm1, ff1, ff2, norm2 in self.enc_blocks:
            attn_out = attn(x, x)
            x = norm1(x + attn_out)
            ff_out = ff2(ff1(x))
            x = norm2(x + ff_out)

        enc_out = x  # Encoder output

        # -------------------
        # Decoder
        # -------------------
        y = self.token_emb(dec_in)
        pos_idx2 = tf.range(tf.shape(y)[1])
        y += self.pos_emb_dec(pos_idx2)

        for self_attn, norm1, enc_attn, norm2, ff1, ff2, norm3 in self.dec_blocks:
            # masked self-attention (prototype, no causal mask yet)
            y = norm1(y + self_attn(y, y))
            # encoder-decoder attention
            y = norm2(y + enc_attn(y, enc_out))
            ff_out = ff2(ff1(y))
            y = norm3(y + ff_out)

        logits = self.final_dense(y)
        return logits
