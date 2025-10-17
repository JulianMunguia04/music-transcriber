import tensorflow as tf
import numpy as np
from pathlib import Path
import random
from src.preprocess import audio_to_mel, events_to_tokens, midi_to_event_sequence

class MaestroDataPipeline:
    def __init__(self, data_path, vocab, batch_size=8, max_mel_len=2048, max_seq_len=1024):
        self.data_path = Path(data_path)
        self.vocab = vocab
        self.batch_size = batch_size
        self.max_mel_len = max_mel_len
        self.max_seq_len = max_seq_len
        
    def get_all_files(self, subset=None):
        """Get list of all files, optionally take subset"""
        all_files = sorted(self.data_path.rglob("*.midi"))
        if subset:
            # Use subset for testing (1%, 10%, etc.)
            subset_size = int(len(all_files) * subset)
            return all_files[:subset_size]
        return all_files
    
    def _process_single_file(self, midi_path):
        """Process single MIDI/WAV pair"""
        wav_path = midi_path.with_suffix('.wav')
        
        # Load and process audio
        mel = audio_to_mel(str(wav_path))
        
        # Load and process MIDI
        events = midi_to_event_sequence(str(midi_path))
        tokens = events_to_tokens(events)
        token_ids = [self.vocab['token_to_id'].get(t, self.vocab['token_to_id']['<UNK>']) 
                    for t in tokens]
        
        return mel, token_ids
    
    def _pad_sequences(self, mel, token_ids):
        """Pad sequences to fixed length"""
        # Pad/crop mel spectrogram
        if mel.shape[0] > self.max_mel_len:
            mel = mel[:self.max_mel_len, :]
        else:
            pad_width = [(0, self.max_mel_len - mel.shape[0]), (0, 0)]
            mel = np.pad(mel, pad_width, mode='constant')
        
        # Pad token sequence
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[:self.max_seq_len]
        else:
            token_ids = token_ids + [self.vocab['token_to_id']['<PAD>']] * (self.max_seq_len - len(token_ids))
        
        return mel, token_ids
    
    def create_dataset(self, subset=None, shuffle=True):
        """Create tf.data.Dataset"""
        files = self.get_all_files(subset)
        print(f"Creating dataset with {len(files)} files")
        
        def generator():
            for midi_file in files:
                try:
                    mel, token_ids = self._process_single_file(midi_file)
                    mel, token_ids = self._pad_sequences(mel, token_ids)
                    yield mel.astype(np.float32), np.array(token_ids, dtype=np.int32)
                except Exception as e:
                    print(f"Error processing {midi_file}: {e}")
                    continue
        
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(self.max_mel_len, 80), dtype=tf.float32),
                tf.TensorSpec(shape=(self.max_seq_len,), dtype=tf.int32)
            )
        )
        
        if shuffle:
            dataset = dataset.shuffle(1000)
        
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset