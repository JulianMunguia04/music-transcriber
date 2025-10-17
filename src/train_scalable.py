import tensorflow as tf
import pickle
from pathlib import Path
import time
from datetime import datetime
from src import config

from src.model_scalable import ScalableTransformer
from src.data_pipeline import MaestroDataPipeline

def setup_training():
    """Setup training with scalable configuration"""
    
    # Training scale options
    SCALE_CONFIGS = {
        'debug': {
            'subset': 0.001,  # 0.1% of data ~ 100MB
            'batch_size': 4,
            'epochs': 2,
            'model_size': 'small'
        },
        'small': {
            'subset': 0.01,   # 1% of data ~ 1GB  
            'batch_size': 8,
            'epochs': 10,
            'model_size': 'small'
        },
        'medium': {
            'subset': 0.1,    # 10% of data ~ 10GB
            'batch_size': 16, 
            'epochs': 25,
            'model_size': 'medium'
        },
        'large': {
            'subset': 0.5,    # 50% of data ~ 50GB
            'batch_size': 16,
            'epochs': 50, 
            'model_size': 'large'
        },
        'full': {
            'subset': None,   # 100% of data
            'batch_size': 16,
            'epochs': 100,
            'model_size': 'large'
        }
    }
    
    MODEL_SIZES = {
        'small': {'d_model': 256, 'num_heads': 4, 'ff_dim': 1024, 'enc_layers': 4, 'dec_layers': 4},
        'medium': {'d_model': 512, 'num_heads': 8, 'ff_dim': 2048, 'enc_layers': 6, 'dec_layers': 6},
        'large': {'d_model': 768, 'num_heads': 12, 'ff_dim': 3072, 'enc_layers': 12, 'dec_layers': 12}
    }
    
    # CHOOSE YOUR SCALE:
    scale = 'small'  # Start with small!
    
    config = SCALE_CONFIGS[scale]
    model_config = MODEL_SIZES[config['model_size']]
    
    print(f"🚀 Training Scale: {scale.upper()}")
    print(f"📊 Data: {config['subset']*100 if config['subset'] else 100}% of dataset")
    print(f"🎯 Model: {config['model_size']} size")
    print(f"📈 Batch size: {config['batch_size']}")
    print(f"⏱️  Epochs: {config['epochs']}")
    
    return config, model_config

def train_scalable():
    """Main training function"""
    
    # Load vocabulary (build this first from full dataset)
    with open('full_vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    
    vocab_size = len(vocab['token_to_id'])
    print(f"Vocabulary size: {vocab_size}")
    
    # Setup training scale
    train_config, model_config = setup_training()
    
    # Create model
    model = ScalableTransformer(
        vocab_size=vocab_size,
        **model_config
    )
    
    # Build model with dummy data
    dummy_mel = tf.random.normal((1, 2048, 80))
    dummy_tokens = tf.ones((1, 100), dtype=tf.int32)
    _ = model([dummy_mel, dummy_tokens])
    
    print("✅ Model built successfully")
    model.summary()
    
    # Create data pipeline
    pipeline = MaestroDataPipeline(
        data_path=config.MAESTRO_PATH_FULL,
        vocab=vocab,
        batch_size=train_config['batch_size']
    )
    
    # Create datasets
    train_dataset = pipeline.create_dataset(subset=train_config['subset'])
    # val_dataset = pipeline.create_dataset(subset=0.01)  # 1% for validation
    
    # Setup optimizer and loss
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-4,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9
    )
    
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    # Training metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    
    @tf.function
    def train_step(mel_batch, token_batch):
        dec_input = token_batch[:, :-1]
        targets = token_batch[:, 1:]
        
        with tf.GradientTape() as tape:
            logits = model([mel_batch, dec_input], training=True)
            loss = loss_fn(targets, logits)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        gradients = [tf.clip_by_norm(g, 1.0) for g in gradients]
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        train_loss(loss)
        train_accuracy(targets, logits)
        
        return loss
    
    # Training loop
    print("🎵 Starting training...")
    
    for epoch in range(train_config['epochs']):
        start_time = time.time()
        train_loss.reset_state()
        train_accuracy.reset_state()
        
        for step, (mel_batch, token_batch) in enumerate(train_dataset):
            loss = train_step(mel_batch, token_batch)
            
            if step % 100 == 0:
                print(f"Epoch {epoch+1}, Step {step}: Loss: {loss:.4f}, Acc: {train_accuracy.result():.4f}")
        
        epoch_time = time.time() - start_time
        print(f"✅ Epoch {epoch+1} completed in {epoch_time:.2f}s - "
              f"Loss: {train_loss.result():.4f}, Accuracy: {train_accuracy.result():.4f}")
    
    # Save model
    model.save_weights(f"scalable_model_{train_config['model_size']}.weights.h5")
    print("💾 Model saved!")

if __name__ == "__main__":
    train_scalable()