from collections import Counter
from pathlib import Path
import pickle
from src import config
from src.preprocess import midi_to_event_sequence, events_to_tokens

def build_full_vocab():
    """Build vocabulary from entire MAESTRO dataset"""
    maestro_path = Path(config.MAESTRO_PATH_FULL)
    all_files = sorted(maestro_path.rglob("*.midi"))
    
    print(f"Building vocabulary from {len(all_files)} files...")
    
    all_tokens = []
    for i, midi_file in enumerate(all_files):
        if i % 500 == 0:
            print(f"Processed {i}/{len(all_files)} files...")
        
        try:
            events = midi_to_event_sequence(str(midi_file))
            tokens = events_to_tokens(events)
            all_tokens.extend(tokens)
        except Exception as e:
            continue
    
    # Build vocabulary
    counter = Counter(all_tokens)
    print(f"Total unique tokens: {len(counter)}")
    
    # Filter rare tokens
    min_frequency = 10  # Only keep tokens that appear at least 10 times
    vocab_tokens = [token for token, count in counter.items() if count >= min_frequency]
    
    vocab = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"] + sorted(vocab_tokens)
    token_to_id = {t: i for i, t in enumerate(vocab)}
    id_to_token = {i: t for t, i in token_to_id.items()}
    
    print(f"Final vocabulary size: {len(vocab)}")
    print(f"Most common tokens: {counter.most_common(20)}")
    
    # Save
    with open('full_vocab.pkl', 'wb') as f:
        pickle.dump({
            'token_to_id': token_to_id,
            'id_to_token': id_to_token,
            'token_frequencies': dict(counter.most_common(1000)),
            'total_tokens': len(all_tokens)
        }, f)
    
    print("✅ Full vocabulary built and saved!")

if __name__ == "__main__":
    build_full_vocab()