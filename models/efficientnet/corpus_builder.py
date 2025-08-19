import json
import os
from sklearn.model_selection import train_test_split

# Paths
CORPUS_FILE = os.path.join(os.path.dirname(__file__), 'sample_corpus.txt')
TRAIN_FILE = os.path.join(os.path.dirname(__file__), 'train_corpus.txt')
VAL_FILE = os.path.join(os.path.dirname(__file__), 'val_corpus.txt')
VOCAB_FILE = os.path.join(os.path.dirname(__file__), 'vocab.json')

# Parameters
MAX_SEQ_LEN = 200  # max characters per sequence
TEST_SIZE = 0.2    # 80/20 split


def build_vocab(lines):
    """
    Build char-to-index mapping from list of text lines.
    """
    chars = sorted(set(''.join(lines)))
    vocab = {ch: i for i, ch in enumerate(chars)}
    vocab['<unk>'] = len(vocab)
    return vocab


def load_corpus(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def chunk_sequences(lines, max_len):
    """
    Split lines into fixed-length character sequences.
    """
    seqs = []
    for line in lines:
        for i in range(0, len(line), max_len):
            chunk = line[i:i + max_len]
            if len(chunk) > 1:
                seqs.append(chunk)
    return seqs


def main():
    # Load and preprocess
    lines = load_corpus(CORPUS_FILE)
    seqs = chunk_sequences(lines, MAX_SEQ_LEN)
    print(f"Total sequences after chunking: {len(seqs)}")

    # Split into train/validation
    train_seqs, val_seqs = train_test_split(seqs, test_size=TEST_SIZE, random_state=42)
    print(f"Train sequences: {len(train_seqs)}, Validation sequences: {len(val_seqs)}")

    # Build and save vocab
    vocab = build_vocab(train_seqs)
    with open(VOCAB_FILE, 'w', encoding='utf-8') as vf:
        json.dump(vocab, vf, ensure_ascii=False, indent=2)
    print(f"Vocab size: {len(vocab)} characters, saved to {VOCAB_FILE}")

    # Write sequences
    with open(TRAIN_FILE, 'w', encoding='utf-8') as tf:
        for seq in train_seqs:
            tf.write(seq + '\n')
    with open(VAL_FILE, 'w', encoding='utf-8') as vf:
        for seq in val_seqs:
            vf.write(seq + '\n')
    print(f"Train/val files written: {TRAIN_FILE}, {VAL_FILE}")


if __name__ == '__main__':
    main()
