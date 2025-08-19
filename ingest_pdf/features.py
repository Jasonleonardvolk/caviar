import re
import numpy as np
from collections import Counter
from typing import List, Tuple, Sequence

def _tokenise(text: str) -> List[str]:
    """Ultra-light lexical split."""
    return re.findall(r"[a-z]{3,}|\d+", text.lower())

def build_feature_matrix(blocks: Sequence[str], vocab_size: int = 1000) -> Tuple[np.ndarray, List[str]]:
    """Return (n_blocks, vocab_size) normalised count matrix + vocab list."""
    corpus = Counter(w for blk in blocks for w in _tokenise(blk))
    vocab = [w for w, _ in corpus.most_common(vocab_size)]
    idx = {w: i for i, w in enumerate(vocab)}
    mat = np.zeros((len(blocks), len(vocab)), dtype=np.float32)
    for r, blk in enumerate(blocks):
        for w in _tokenise(blk):
            if w in idx:
                mat[r, idx[w]] += 1
    mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
    return mat, vocab

def compute_block_tfidf_embedding(block: str, corpus: Counter, vocab_size: int = 1000) -> np.ndarray:
    """Compute TF-IDF vector for a block given a corpus."""
    vocab = [w for w, _ in corpus.most_common(vocab_size)]
    idx = {w: i for i, w in enumerate(vocab)}
    block_counts = Counter(_tokenise(block))
    tfidf = np.zeros(len(vocab), dtype=np.float32)
    for w, count in block_counts.items():
        if w in idx:
            tf = count / len(block_counts)
            idf = np.log(len(corpus) / (1 + corpus[w]))
            tfidf[idx[w]] = tf * idf
    tfidf /= np.linalg.norm(tfidf) + 1e-8
    return tfidf
