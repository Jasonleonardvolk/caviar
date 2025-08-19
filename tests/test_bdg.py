import pytest
import numpy as np
from python.core.bdg_solver import assemble_bdg, compute_spectrum, extract_lyapunov_exponents, analyze_stability

def test_bdg_symmetry():
    """Test eigenvalue symmetry: if ω is eigenvalue, so is -ω*"""
    # Create test soliton
    x = np.linspace(-10, 10, 128)
    X, Y = np.meshgrid(x, x)
    psi0 = np.tanh(X/2) * np.exp(1j * 0)  # 1D dark soliton
    
    # Build BdG
    H = assemble_bdg(psi0)
    eigenvalues, _ = compute_spectrum(H, k=16)
    
    # Check symmetry (with tolerance for numerical errors)
    for omega in eigenvalues:
        # For each eigenvalue, there should be a -omega* 
        found_conjugate = False
        for other in eigenvalues:
            if np.abs(other + np.conj(omega)) < 1e-10:
                found_conjugate = True
                break
        assert found_conjugate, f"No conjugate pair found for eigenvalue {omega}"

def test_stable_soliton():
    """Test that a stationary dark soliton is stable"""
    # Create stable dark soliton
    x = np.linspace(-20, 20, 64)
    X, Y = np.meshgrid(x, x)
    psi0 = np.tanh(X/4.0)  # Wide, stable soliton
    
    # Analyze stability
    H = assemble_bdg(psi0, g=1.0, dx=x[1]-x[0])
    eigenvalues, _ = compute_spectrum(H, k=8)
    lyapunov = extract_lyapunov_exponents(eigenvalues)
    
    stability = analyze_stability(lyapunov)
    
    # Should be stable
    assert stability['stable'] == True
    assert stability['max_lyapunov'] <= 1e-10  # Numerical tolerance
    assert stability['unstable_modes'] == 0

def test_laplacian_matrix():
    """Test that Laplacian matrix is correct"""
    from python.core.bdg_solver import build_laplacian_matrix
    
    # Small test case
    shape = (3, 3)
    dx = 1.0
    L = build_laplacian_matrix(shape, dx)
    
    # Check matrix properties
    assert L.shape == (9, 9)  # Flattened 3x3 grid
    
    # Check that it's symmetric
    assert np.allclose(L.toarray(), L.toarray().T)
    
    # Check diagonal elements (should be -4/dx^2)
    diag = L.diagonal()
    assert np.allclose(diag, -4.0/dx**2)

def test_lyapunov_extraction():
    """Test Lyapunov exponent extraction"""
    # Create test eigenvalues with known imaginary parts
    test_eigenvalues = np.array([
        0.1 + 0.2j,   # Unstable mode
        0.3 - 0.1j,   # Stable mode
        0.0 + 0.0j,   # Neutral mode
        -0.2 + 0.15j  # Unstable mode
    ])
    
    lyapunov = extract_lyapunov_exponents(test_eigenvalues)
    
    # Check extraction
    assert np.allclose(lyapunov[0], 0.2)
    assert np.allclose(lyapunov[1], -0.1)
    assert np.allclose(lyapunov[2], 0.0)
    assert np.allclose(lyapunov[3], 0.15)
    
    # Check stability analysis
    stability = analyze_stability(lyapunov)
    assert stability['stable'] == False
    assert stability['max_lyapunov'] == 0.2
    assert stability['unstable_modes'] == 2

def test_gpu_cpu_consistency():
    """Test that GPU and CPU give same results"""
    # Skip if no GPU
    try:
        import cupy
    except ImportError:
        pytest.skip("GPU not available")
    
    # Create test soliton
    x = np.linspace(-10, 10, 32)
    X, Y = np.meshgrid(x, x)
    psi0 = np.tanh(X/2)
    
    # Force CPU computation
    import python.core.bdg_solver as bdg
    original_gpu = bdg.GPU_AVAILABLE
    
    try:
        # CPU computation
        bdg.GPU_AVAILABLE = False
        H_cpu = assemble_bdg(psi0)
        
        # GPU computation
        bdg.GPU_AVAILABLE = True
        H_gpu = assemble_bdg(psi0)
        
        # Compare (allowing for small numerical differences)
        assert np.allclose(H_cpu.toarray(), H_gpu.toarray(), rtol=1e-10)
        
    finally:
        bdg.GPU_AVAILABLE = original_gpu

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
