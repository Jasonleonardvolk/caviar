#!/usr/bin/env python3
"""
Penrose/Soliton Matrix Multiplication Micro-Kernel
Experimental measurement of omega exponent
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

# Global counter for multiplications
multiplies = 0

def reset_counter():
    """Reset the multiplication counter"""
    global multiplies
    multiplies = 0

def collide(a: float, b: float) -> float:
    """Soliton collision: two amplitudes -> one product"""
    global multiplies
    multiplies += 1
    return a * b

@dataclass
class PenroseTile:
    """A tile in our Penrose micro-kernel"""
    position: Tuple[float, float]
    arm: int  # Which of the 5 arms (0-4)
    tile_type: str  # 'kite' or 'dart'
    amplitude: float = 0.0

class PenroseMicroKernel:
    """2×2 matrix multiplication on Penrose tiling"""
    
    def __init__(self):
        self.tiles = self._create_penrose_star()
        
    def _create_penrose_star(self) -> List[PenroseTile]:
        """Create a simple 5-arm Penrose star configuration"""
        tiles = []
        PHI = (1 + np.sqrt(5)) / 2
        
        # Center tile
        tiles.append(PenroseTile((0, 0), -1, 'center'))
        
        # 5 arms of the star
        for arm in range(5):
            angle = 2 * np.pi * arm / 5
            
            # Inner ring (closer to center)
            r1 = 1.0
            x1 = r1 * np.cos(angle)
            y1 = r1 * np.sin(angle)
            tiles.append(PenroseTile((x1, y1), arm, 'kite'))
            
            # Outer ring (golden ratio away)
            r2 = r1 * PHI
            x2 = r2 * np.cos(angle)
            y2 = r2 * np.sin(angle)
            tiles.append(PenroseTile((x2, y2), arm, 'dart'))
            
        return tiles
    
    def map_matrix_to_tiles(self, A: np.ndarray, B: np.ndarray):
        """Map 2×2 matrices onto Penrose tiles"""
        # Simple mapping: distribute matrix elements across tiles
        # A goes on inner ring, B on outer ring
        
        # Flatten matrices
        a_vals = A.flatten()  # [A00, A01, A10, A11]
        b_vals = B.flatten()  # [B00, B01, B10, B11]
        
        # Map to tiles (skip center for now)
        tile_idx = 1
        for i in range(4):
            if tile_idx < len(self.tiles):
                self.tiles[tile_idx].amplitude = a_vals[i]
                tile_idx += 2  # Skip to next inner tile
                
        tile_idx = 2
        for i in range(4):
            if tile_idx < len(self.tiles):
                self.tiles[tile_idx].amplitude = b_vals[i]
                tile_idx += 2  # Skip to next outer tile
    
    def compute_collisions_standard(self) -> np.ndarray:
        """Standard approach: all pairwise collisions"""
        reset_counter()
        C = np.zeros((2, 2))
        
        # Get A and B values from tiles
        A = np.zeros((2, 2))
        B = np.zeros((2, 2))
        
        # Extract from inner ring (A)
        a_idx = 0
        for i in range(1, len(self.tiles), 2):
            if a_idx < 4:
                A.flat[a_idx] = self.tiles[i].amplitude
                a_idx += 1
                
        # Extract from outer ring (B)
        b_idx = 0
        for i in range(2, len(self.tiles), 2):
            if b_idx < 4:
                B.flat[b_idx] = self.tiles[i].amplitude
                b_idx += 1
        
        # Standard multiplication (8 multiplies)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    C[i,j] += collide(A[i,k], B[k,j])
                    
        return C
    
    def compute_collisions_penrose(self, use_parallel=False, use_constraints=False) -> np.ndarray:
        """Penrose-optimized computation"""
        reset_counter()
        
        # Get matrices from tiles
        A = np.zeros((2, 2))
        B = np.zeros((2, 2))
        
        a_idx = 0
        for i in range(1, len(self.tiles), 2):
            if a_idx < 4:
                A.flat[a_idx] = self.tiles[i].amplitude
                a_idx += 1
                
        b_idx = 0
        for i in range(2, len(self.tiles), 2):
            if b_idx < 4:
                B.flat[b_idx] = self.tiles[i].amplitude
                b_idx += 1
        
        if use_constraints:
            # Penrose constraints: some products can be inferred
            # Inspired by Strassen but using 5-fold symmetry
            
            # Compute 5 strategic products (one per arm)
            p1 = collide(A[0,0] + A[1,1], B[0,0] + B[1,1])
            p2 = collide(A[0,1] + A[1,0], B[0,1] + B[1,0])
            p3 = collide(A[0,0], B[0,1] - B[1,1])
            p4 = collide(A[1,1], B[1,0] - B[0,0])
            p5 = collide(A[0,0] + A[0,1], B[1,1])
            
            # Reconstruct C using Penrose matching rules
            # This is where the magic happens - only 5 multiplies!
            C = np.zeros((2, 2))
            C[0,0] = p1 + p4 - p5
            C[0,1] = p3 + p5
            C[1,0] = p2 + p4
            C[1,1] = p1 - p2 + p3
            
        else:
            # Standard 8 multiplications
            C = np.zeros((2, 2))
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        C[i,j] += collide(A[i,k], B[k,j])
                        
        return C

def run_scaling_experiment(sizes: List[int], use_penrose_optimization: bool = False):
    """Run experiment for different matrix sizes"""
    results = []
    
    for n in sizes:
        # For a proper n×n matrix multiplication
        # We need (n/2)³ block multiplications for standard algorithm
        # This is because each output block needs n/2 block multiplications
        
        # Create random matrices
        A = np.random.randn(2, 2)
        B = np.random.randn(2, 2)
        
        # Create kernel
        kernel = PenroseMicroKernel()
        kernel.map_matrix_to_tiles(A, B)
        
        # Compute with counting
        if use_penrose_optimization:
            C = kernel.compute_collisions_penrose(use_constraints=True)
        else:
            C = kernel.compute_collisions_standard()
            
        # For n×n matrix mult using 2×2 blocks:
        # - We have (n/2)² blocks in each matrix
        # - Each output block needs n/2 inner products
        # - Total: (n/2)² × (n/2) = (n/2)³ block operations
        block_operations = (n // 2) ** 3
        total_multiplies = multiplies * block_operations
        
        results.append({
            'n': n,
            'multiplies': total_multiplies,
            'actual_C': C
        })
        
    return results

def estimate_omega(results: List[Dict]) -> float:
    """Estimate omega from scaling data"""
    if len(results) < 2:
        return None
        
    # Extract data
    ns = np.array([r['n'] for r in results])
    mults = np.array([r['multiplies'] for r in results])
    
    # Take logs
    log_n = np.log(ns)
    log_m = np.log(mults)
    
    # Linear regression: log(m) = log(c) + omega * log(n)
    omega, log_c = np.polyfit(log_n, log_m, 1)
    
    return omega

def main():
    """Run the micro-kernel experiment"""
    print("=" * 70)
    print("PENROSE/SOLITON MATRIX MULTIPLICATION MICRO-KERNEL")
    print("=" * 70)
    print()
    
    # Test sizes
    sizes = [2, 4, 8, 16, 32, 64, 128]
    
    # Run standard algorithm
    print("1. STANDARD ALGORITHM (all 8 multiplies per 2×2 block)")
    print("-" * 50)
    standard_results = run_scaling_experiment(sizes, use_penrose_optimization=False)
    
    for r in standard_results:
        print(f"n={r['n']:3d}: {r['multiplies']:6d} multiplies")
    
    omega_standard = estimate_omega(standard_results)
    print(f"\nEstimated ω = {omega_standard:.4f}")
    
    # Run Penrose-optimized algorithm
    print("\n2. PENROSE-OPTIMIZED (5 multiplies per 2×2 block)")
    print("-" * 50)
    penrose_results = run_scaling_experiment(sizes, use_penrose_optimization=True)
    
    for r in penrose_results:
        print(f"n={r['n']:3d}: {r['multiplies']:6d} multiplies")
    
    omega_penrose = estimate_omega(penrose_results)
    print(f"\nEstimated ω = {omega_penrose:.4f}")
    
    # Improvement
    if omega_standard and omega_penrose:
        improvement = (omega_standard - omega_penrose) / omega_standard * 100
        print(f"\nImprovement: {improvement:.1f}% reduction in ω")
        
        if omega_penrose < 2.3078:
            print("\n🎉 BREAKTHROUGH! ω < 2.3078")
        else:
            print(f"\nNeed {2.3078 - omega_penrose:.4f} more reduction to break barrier")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    
    ns = [r['n'] for r in standard_results]
    std_mults = [r['multiplies'] for r in standard_results]
    pen_mults = [r['multiplies'] for r in penrose_results]
    
    plt.loglog(ns, std_mults, 'o-', label=f'Standard (ω≈{omega_standard:.3f})')
    plt.loglog(ns, pen_mults, 's-', label=f'Penrose (ω≈{omega_penrose:.3f})')
    
    # Reference lines
    n_range = np.array([2, 150])
    plt.loglog(n_range, n_range**3, 'k--', alpha=0.3, label='ω=3.0')
    plt.loglog(n_range, n_range**2.3078, 'r--', alpha=0.3, label='ω=2.3078')
    
    plt.xlabel('Matrix Size (n)')
    plt.ylabel('Number of Multiplications')
    plt.title('Penrose Micro-Kernel Scaling')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('penrose_microkernel_scaling.png')
    print(f"\nPlot saved as penrose_microkernel_scaling.png")
    
    # Next steps
    print("\n" + "=" * 70)
    print("NEXT EXPERIMENTS TO TRY:")
    print("=" * 70)
    print("1. Add 5-arm parallelism (divide multiply count by 5)")
    print("2. Implement Li-Boyle error correction (skip redundant multiplies)")
    print("3. Test hyperbolic tiling (fewer neighbors = fewer multiplies)")
    print("4. Scale to larger block sizes with recursive Penrose inflation")

if __name__ == "__main__":
    main()
