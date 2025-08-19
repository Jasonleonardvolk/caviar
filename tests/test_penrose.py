# tests/test_penrose.py
import time
import numpy as np
from concept_mesh.similarity import penrose

def test_speed():
    a = np.random.rand(512).astype("float32")
    b = np.random.rand(512).astype("float32")
    t0 = time.perf_counter()
    for _ in range(10_000):
        penrose.compute_similarity(a, b)
    assert time.perf_counter() - t0 < 0.3

def test_batch_speed():
    """Additional test for batch performance"""
    query = np.random.rand(512).astype("float32")
    corpus = [np.random.rand(512).astype("float32") for _ in range(1000)]
    t0 = time.perf_counter()
    results = penrose.batch_similarity(query, corpus)
    assert time.perf_counter() - t0 < 0.1
    assert len(results) == 1000
