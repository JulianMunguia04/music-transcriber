import tensorflow as tf
from tensorflow.keras import layers
import math

class LearnablePositionalEncoding(layers.Layer):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.position_embeddings = layers.Embedding(input_dim=max_len, output_dim=d_model)
        
    def call(self, x):
        seq_len = tf.shape(x)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        positions = tf.minimum(positions, self.max_len - 1)
        pos_encoding = self.position_embeddings(positions)
        return x + pos_encoding

class AudioCNNFrontend(layers.Layer):
    """Fixed CNN frontend that maintains sequence length"""
    def __init__(self, d_model):
        super().__init__()
        # Use convolution without pooling to maintain sequence length
        self.conv_layers = [
            layers.Conv1D(128, 5, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv1D(256, 5, activation='relu', padding='same'),
            layers.BatchNormalization(), 
            layers.Conv1D(d_model, 5, activation='relu', padding='same'),
            layers.BatchNormalization(),
        ]
        
    def call(self, x):
        # x: (batch, time, 80)
        for layer in self.conv_layers:
            x = layer(x)
        return x  # Output: (batch, time, d_model) - same time dimension!

class TransformerEncoderLayer(layers.Layer):
    """Proper encoder layer as a Layer instead of Model"""
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model//num_heads, dropout=dropout
        )
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(d_model)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
        
    def call(self, x, training=False):
        # Self attention
        attn_output = self.self_attention(x, x, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        # Feed forward
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

class TransformerDecoderLayer(layers.Layer):
    """Proper decoder layer with causal masking"""
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model//num_heads, dropout=dropout
        )
        self.cross_attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model//num_heads, dropout=dropout
        )
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(d_model)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
        self.dropout3 = layers.Dropout(dropout)
        
    def call(self, x, enc_output, training=False, look_ahead_mask=None):
        # Self attention with causal masking
        self_attn = self.self_attention(
            x, x, attention_mask=look_ahead_mask, training=training
        )
        self_attn = self.dropout1(self_attn, training=training)
        out1 = self.layernorm1(x + self_attn)
        
        # Cross attention
        cross_attn = self.cross_attention(
            out1, enc_output, training=training
        )
        cross_attn = self.dropout2(cross_attn, training=training)
        out2 = self.layernorm2(out1 + cross_attn)
        
        # Feed forward
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output)
        return out3

class ScalableTransformer(tf.keras.Model):
    def __init__(self, vocab_size, d_model=512, num_heads=8, ff_dim=2048,
                 enc_layers=6, dec_layers=6, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Audio processing - FIXED
        self.audio_projection = layers.Dense(d_model)  # Simple projection first
        self.audio_frontend = AudioCNNFrontend(d_model)
        
        # Encoder
        self.enc_pos_encoding = LearnablePositionalEncoding(d_model)
        self.enc_dropout = layers.Dropout(dropout)
        self.encoder_layers = [
            TransformerEncoderLayer(d_model, num_heads, ff_dim, dropout)
            for _ in range(enc_layers)
        ]
        
        # Decoder  
        self.token_emb = layers.Embedding(vocab_size, d_model)
        self.dec_pos_encoding = LearnablePositionalEncoding(d_model)
        self.dec_dropout = layers.Dropout(dropout)
        self.decoder_layers = [
            TransformerDecoderLayer(d_model, num_heads, ff_dim, dropout)
            for _ in range(dec_layers)
        ]
        
        # Output
        self.output_layer = layers.Dense(vocab_size)
        
    def create_look_ahead_mask(self, size):
        """Create causal mask for decoder"""
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq_len, seq_len)
    
    def call(self, inputs, training=False):
        enc_inputs, dec_inputs = inputs
        
        # ===== ENCODER =====
        # Project audio to d_model dimension
        x = self.audio_projection(enc_inputs)  # (batch, time, 80) -> (batch, time, d_model)
        x = self.audio_frontend(x)  # Process with CNN
        x = self.enc_pos_encoding(x)
        x = self.enc_dropout(x, training=training)
        
        # Encoder layers
        for layer in self.encoder_layers:
            x = layer(x, training=training)
        
        # ===== DECODER =====
        # Create causal mask
        dec_seq_len = tf.shape(dec_inputs)[1]
        look_ahead_mask = self.create_look_ahead_mask(dec_seq_len)
        
        y = self.token_emb(dec_inputs)  # (batch, seq_len) -> (batch, seq_len, d_model)
        y = self.dec_pos_encoding(y)
        y = self.dec_dropout(y, training=training)
        
        # Decoder layers
        for layer in self.decoder_layers:
            y = layer(y, x, training=training, look_ahead_mask=look_ahead_mask)
        
        # Output
        logits = self.output_layer(y)  # (batch, seq_len, vocab_size)
        return logits
    