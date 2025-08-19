import pytest
from extraction import extract_concepts_universal

def test_extract_simple_text():
    text = "Deep learning and neural networks are popular AI methods."
    concepts = extract_concepts_universal(text)
    names = [c['name'].lower() for c in concepts]
    assert any("deep learning" in n for n in names)
    assert any("neural network" in n or "neural networks" in n for n in names)

def test_empty_text():
    assert extract_concepts_universal("") == []

def test_stopwords_are_filtered():
    text = "The the the the the and and and."
    concepts = extract_concepts_universal(text)
    assert concepts == []
