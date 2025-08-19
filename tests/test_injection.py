import pytest
from injection import inject_concepts_into_mesh

def test_injection_basic(monkeypatch):
    calls = []
    def fake_add_concept(**kwargs):
        calls.append(kwargs)
    # monkeypatch Prajna API here
    # monkeypatch.setattr("prajna_mesh.add_concept", fake_add_concept)
    result = inject_concepts_into_mesh([{"name": "Foo"}])
    assert result["injected"] == 1
