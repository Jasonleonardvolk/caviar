# Development Setup for Penrose Engine
# PowerShell script for setting up development environment

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Penrose Engine Development Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Navigate to project
Set-Location "C:\Users\jason\Desktop\tori\kha\concept_mesh\penrose_rs"

# Create development virtual environment
Write-Host "Creating development virtual environment..." -ForegroundColor Yellow
python -m venv ..\..\venv_penrose_dev

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& "..\..\venv_penrose_dev\Scripts\Activate.ps1"

# Install development dependencies
Write-Host "`nInstalling development dependencies..." -ForegroundColor Yellow
pip install --upgrade pip setuptools wheel
pip install maturin numpy pytest black mypy ruff

# Install in development mode
Write-Host "`nInstalling Penrose Engine in development mode..." -ForegroundColor Yellow
maturin develop

# Create VS Code settings
Write-Host "`nCreating VS Code settings..." -ForegroundColor Yellow
$vscodeSettings = @'
{
    "python.defaultInterpreterPath": "${workspaceFolder}/venv_penrose_dev/Scripts/python.exe",
    "rust-analyzer.cargo.features": ["pyo3/extension-module"],
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "[rust]": {
        "editor.formatOnSave": true
    },
    "[python]": {
        "editor.formatOnSave": true
    }
}
'@

if (-not (Test-Path ".vscode")) {
    New-Item -ItemType Directory -Name ".vscode"
}
$vscodeSettings | Out-File -FilePath ".vscode\settings.json" -Encoding UTF8

# Create development README
Write-Host "`nCreating development README..." -ForegroundColor Yellow
$devReadme = @'
# Penrose Engine Development

## Quick Commands

### Build and Test
```bash
# Build in development mode (fast, unoptimized)
maturin develop

# Build in release mode (optimized)
maturin build --release

# Run tests
pytest tests/

# Format code
black .
cargo fmt
```

### Development Workflow

1. Make changes to `src/lib.rs`
2. Run `maturin develop` to rebuild
3. Test your changes in Python
4. Run tests with `pytest`

### Adding New Functions

1. Add Rust function in `src/lib.rs`:
```rust
#[pyfunction]
fn my_new_function(param: i32) -> PyResult<i32> {
    Ok(param * 2)
}
```

2. Register in module:
```rust
#[pymodule]
fn penrose_engine_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(my_new_function, m)?)?;
    Ok(())
}
```

3. Rebuild with `maturin develop`

### Performance Profiling

```python
import cProfile
import penrose_engine_rs

cProfile.run('penrose_engine_rs.your_function()')
```

### Memory Debugging

Set environment variable:
```
$env:RUST_BACKTRACE="full"
```
'@

$devReadme | Out-File -FilePath "DEVELOPMENT.md" -Encoding UTF8

# Create test directory and sample test
Write-Host "`nCreating test structure..." -ForegroundColor Yellow
if (-not (Test-Path "tests")) {
    New-Item -ItemType Directory -Name "tests"
}

$sampleTest = @'
import pytest
import penrose_engine_rs

def test_engine_initialization():
    """Test engine initialization"""
    result = penrose_engine_rs.initialize_engine(
        max_threads=2,
        cache_size_mb=256,
        enable_gpu=False,
        precision="float32"
    )
    assert result["success"] == True
    assert result["thread_count"] == 2

def test_similarity_computation():
    """Test similarity computation"""
    v1 = [1.0, 2.0, 3.0]
    v2 = [4.0, 5.0, 6.0]
    sim = penrose_engine_rs.compute_similarity(v1, v2)
    assert 0.974 < sim < 0.975  # Expected cosine similarity

def test_batch_similarity():
    """Test batch similarity computation"""
    query = [1.0, 0.0, 0.0]
    corpus = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    sims = penrose_engine_rs.batch_similarity(query, corpus)
    assert len(sims) == 3
    assert abs(sims[0] - 1.0) < 0.001  # First vector should have similarity 1
    assert all(abs(s) < 0.001 for s in sims[1:])  # Others should be 0

def test_error_handling():
    """Test error handling for mismatched dimensions"""
    v1 = [1.0, 2.0, 3.0]
    v2 = [4.0, 5.0]  # Different dimension
    with pytest.raises(Exception):
        penrose_engine_rs.compute_similarity(v1, v2)
'@

$sampleTest | Out-File -FilePath "tests\test_penrose.py" -Encoding UTF8

# Create __init__.py for tests
"" | Out-File -FilePath "tests\__init__.py" -Encoding UTF8

# Run initial test
Write-Host "`nRunning initial tests..." -ForegroundColor Yellow
python -m pytest tests/ -v

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Development Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Development environment is ready!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Edit src/lib.rs to add/modify functions" -ForegroundColor White
Write-Host "2. Run 'maturin develop' to rebuild" -ForegroundColor White
Write-Host "3. Test your changes in Python" -ForegroundColor White
Write-Host "4. Run 'pytest tests/' to run tests" -ForegroundColor White
Write-Host ""
Write-Host "VS Code should auto-detect the virtual environment." -ForegroundColor Cyan
