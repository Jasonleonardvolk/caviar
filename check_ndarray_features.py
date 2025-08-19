#!/usr/bin/env python3
"""
Check available features for ndarray-linalg
"""

print("🔍 Checking ndarray-linalg features...")
print("=" * 60)

print("""
According to the ndarray-linalg docs, the available features are:

For version 0.16:
- intel-mkl (Intel MKL backend)
- netlib (Netlib LAPACK backend)  
- openblas (OpenBLAS backend)
- openblas-static (Static OpenBLAS)
- openblas-system (System OpenBLAS)

There is NO "rust" feature!

The solution is to use ndarray WITHOUT linalg for pure Rust:
- Just use ndarray = "0.15" 
- Remove ndarray-linalg entirely
- Or use ndarray-linalg with no backend (for basic operations)
""")

print("\n✅ Let's fix this properly...")
