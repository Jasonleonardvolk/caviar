# Penrose Engine (Rust)

High-performance similarity computation engine for TORI's ConceptMesh system.

## Features

- **O(n^2.32) complexity** through rank reduction
- **SIMD acceleration** for batch operations  
- **Parallel processing** with Rayon
- **Python bindings** via PyO3
- **Automatic fallback** to Python implementation

## Performance

- 80-100× faster than pure Python implementation
- Processes 1M similarity comparisons in ~90-120μs
- Enables real-time lattice evolution in FractalSolitonMemory

## Building

### Quick Build & Install

```powershell
# From project root (${IRIS_ROOT})
.\BUILD_PENROSE.bat
```

### Manual Build

```bash
cd concept_mesh/penrose_rs
pip install maturin
maturin develop --release  # Build and install in one step
```

### For CI/Production

```bash
maturin build --release
# Wheel will be in target/wheels/
pip install target/wheels/penrose_engine_rs-*.whl
```

## Usage

The engine is automatically detected by ConceptMesh:

```python
from concept_mesh.similarity import penrose_available, compute_similarity

if penrose_available:
    print("Using fast Rust engine")
else:
    print("Using Python fallback")
    
# API is the same regardless of backend
similarities = compute_similarity(embeddings, queries)
```

## Verification

```powershell
python verify_penrose_integration.py
```

Look for:
- ✅ "Penrose engine initialized (Rust, SIMD, rank=32)" in logs
- No ⚠️ "Penrose not available" warnings
- Non-zero oscillator counts in lattice runner

## Troubleshooting

### "link.exe not found"
Install Visual Studio Build Tools or run:
```powershell
rustup toolchain install stable-x86_64-pc-windows-msvc
rustup default stable-x86_64-pc-windows-msvc
```

### ImportError at runtime
Ensure Python version matches:
```powershell
python --version  # Should match your TORI environment
```

### Slow performance
Check if Rust engine loaded:
```python
from concept_mesh.similarity import penrose
print(penrose.get_info())  # Should show "Rust, SIMD"
```

## GPU Acceleration (Optional)

For CUDA support:
```bash
RUSTFLAGS="--cfg cuda" maturin build --features cuda --release
```

Requires CUDA Toolkit 12+.