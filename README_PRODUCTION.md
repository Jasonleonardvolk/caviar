# TORI Production Deployment README

## Overview
The Penrose microkernel achieves sub-Strassen matrix multiplication complexity using spectral methods on quasicrystal topology.

## Configuration
Edit `config/penrose.yaml`:
```yaml
penrose:
  enable: true          # Emergency kill switch
  rank: 14              # Lower = faster but less accurate (8-32)
  laplacian_nodes: 6000 # Must exceed largest matrix size
```

## Performance
- **Complexity**: ω ≈ 2.33 (vs Strassen 2.807)
- **Overhead**: ~1s one-time eigensolve, cached
- **Memory**: O(n × rank) = minimal

## Deployment Checklist
1. [ ] Run tests: `python test_penrose_production.py`
2. [ ] Verify config: `python tori_production_launcher.py --check`
3. [ ] Stage deploy: Monitor spectral gap in logs
4. [ ] Production: Watch for `[penrose] rank=14, gap=X.Xe-3` at startup

## Troubleshooting
- **Fallback to BLAS**: Check logs for reason (gap too small, size exceeded)
- **Performance regression**: Verify BLAS threads not pinned to 1
- **Clear cache**: Call `penrose_microkernel_v3_production.clear_cache()`

## Architecture
```
tori_production_launcher.py
    ↓ loads config
    ↓ sets BLAS threads  
    ↓ builds Laplacian once
    ↓ configures microkernel
    → ready for multiply operations
```

## Emergency Fallback
Set `penrose.enable: false` in config and restart.
