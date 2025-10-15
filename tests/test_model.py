# tests/test_model.py
from src.model import SimpleTransformer
import numpy as np

def main():
    vocab_size = 50
    mel = np.random.rand(1, 100, 80).astype(np.float32)  # 100 frames, 80 mels
    dec_in = np.random.randint(0, vocab_size, size=(1, 20))  # 20 tokens

    model = SimpleTransformer(vocab_size)
    logits = model((mel, dec_in))
    print("Logits shape:", logits.shape)  # (1, 20, vocab_size)

if __name__ == "__main__":
    main()
