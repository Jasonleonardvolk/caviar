import os, shutil, tempfile, asyncio, json
import pytest
from pathlib import Path

@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

@pytest.fixture
def tmp_workdir(tmp_path: Path, repo_root: Path, monkeypatch):
    # sandbox logs/adapters so tests never touch prod files
    sandbox = tmp_path / "sandbox"
    (sandbox / "logs" / "inference").mkdir(parents=True, exist_ok=True)
    (sandbox / "adapters").mkdir(parents=True, exist_ok=True)
    (sandbox / "logs" / "mesh").mkdir(parents=True, exist_ok=True)
    (sandbox / "logs" / "chaos").mkdir(parents=True, exist_ok=True)
    (sandbox / "logs" / "errors").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("TORI_LOG_DIR", str(sandbox / "logs"))
    monkeypatch.setenv("TORI_ADAPTER_DIR", str(sandbox / "adapters"))
    return sandbox

@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_mesh():
    """Provide a test mesh object with tracking capabilities."""
    class MockMesh:
        def __init__(self):
            self.nodes = {}
            self.edges = {}
            self.last_updated = None
            self.version = 0
            self.update_count = 0
        
        def update(self, data):
            self.nodes.update(data.get("nodes", {}))
            self.edges.update(data.get("edges", {}))
            self.last_updated = data.get("last_updated")
            self.version += 1
            self.update_count += 1
        
        def to_summary(self):
            return {
                "node_count": len(self.nodes),
                "edge_count": len(self.edges),
                "version": self.version,
                "last_updated": self.last_updated
            }
    
    return MockMesh()
