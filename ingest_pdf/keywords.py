from collections import Counter, defaultdict
from typing import List, Sequence
import re

try:
    # Try absolute import first
    from features import _tokenise
except ImportError:
    # Fallback to relative import
    try:
        # Try absolute import first
        from features import _tokenise
    except ImportError:
        # Fallback to relative import
        from .features import _tokenise
_stop = {"the","and","for","with","that","this","from","into","using","over","such"}

# N-gram extraction (up to trigrams)
def extract_ngrams(words: List[str], min_n: int = 1, max_n: int = 3) -> List[str]:
    ngrams = []
    for n in range(min_n, max_n+1):
        ngrams.extend([" ".join(words[i:i+n]) for i in range(len(words)-n+1)])
    return ngrams

def extract_keywords(
    cluster_blocks: Sequence[str],
    other_blocks: Sequence[str] = (),
    n: int = 3
) -> List[str]:
    # Tokenize and extract n-grams from cluster
    cluster_words = [w for blk in cluster_blocks for w in _tokenise(blk) if w not in _stop]
    cluster_ngrams = extract_ngrams(cluster_words)
    # Tokenize and extract n-grams from other blocks for TF-IDF
    other_words = [w for blk in other_blocks for w in _tokenise(blk) if w not in _stop]
    other_ngrams = extract_ngrams(other_words)
    # Count n-gram frequencies
    cluster_counts = Counter(cluster_ngrams)
    other_counts = Counter(other_ngrams)
    # Compute simple TF-IDF (freq in cluster / (freq in others+1))
    tfidf = {ng: cluster_counts[ng] / (other_counts.get(ng,0)+1) for ng in cluster_counts}
    # Prefer n-grams that appear as section headers (all caps or title-case lines)
    header_candidates = []
    header_pattern = re.compile(r"^[A-Z][A-Za-z\d\s\-:;,\.]+$")
    for blk in cluster_blocks:
        lines = blk.splitlines()
        for line in lines:
            if header_pattern.match(line.strip()) and len(line.strip().split()) <= 8:
                header_candidates.append(line.strip())
    # Rank: headers > high tfidf n-grams > frequent unigrams
    keywords = []
    if header_candidates:
        keywords.extend(header_candidates[:n])
    # Add top tfidf n-grams
    sorted_ngrams = sorted(tfidf.items(), key=lambda kv: kv[1], reverse=True)
    for ng, _ in sorted_ngrams:
        if ng not in keywords and len(keywords) < n:
            keywords.append(ng)
    # Fallback: most common unigrams
    if len(keywords) < n:
        unigram_counts = Counter(cluster_words)
        for w, _ in unigram_counts.most_common(n - len(keywords)):
            if w not in keywords:
                keywords.append(w)
    # Return up to n best
    return keywords[:n]
