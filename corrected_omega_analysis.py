#!/usr/bin/env python3
"""
Penrose/Soliton Matrix Multiplication - CORRECTED ANALYSIS
Properly counts operations for recursive algorithms
"""

import numpy as np
from typing import List, Dict
import matplotlib.pyplot as plt

class MatrixMultiplicationAnalyzer:
    """Analyze matrix multiplication strategies with correct counting"""
    
    def __init__(self):
        self.memo = {}  # For memoization of recursive calls
        
    def naive_multiply_count(self, n: int) -> int:
        """Standard O(n³) algorithm - CORRECT"""
        return n * n * n
    
    def strassen_multiply_count(self, n: int) -> int:
        """Strassen's algorithm: T(n) = 7*T(n/2) - CORRECT"""
        if n <= 1:
            return 1
        # 7 recursive calls on n/2 sized problems
        return 7 * self.strassen_multiply_count(n // 2)
    
    def penrose_basic_multiply_count(self, n: int) -> int:
        """Penrose with 5 multiplies per 2×2 block - CORRECTED
        If we can truly do 2×2 multiplication with 5 multiplies instead of 8,
        then T(n) = 5*T(n/2) for the recursive algorithm
        """
        if n <= 2:
            return 5  # Base case: 5 multiplies for 2×2
        # 5 recursive calls on n/2 sized problems
        return 5 * self.penrose_basic_multiply_count(n // 2)
    
    def hypothetical_penrose_count(self, n: int, multiplies_per_block: int = 5) -> int:
        """General recursive formula: T(n) = m*T(n/2) where m = multiplies per block"""
        if n <= 2:
            return multiplies_per_block
        return multiplies_per_block * self.hypothetical_penrose_count(n // 2, multiplies_per_block)
    
    def compute_omega(self, counts: List[int], sizes: List[int]) -> float:
        """Compute omega from operation counts"""
        if len(sizes) < 2:
            return None
        
        # Use log-log regression
        log_n = np.log(sizes)
        log_counts = np.log(counts)
        omega, log_c = np.polyfit(log_n, log_counts, 1)
        return omega
    
    def theoretical_omega(self, multiplies_per_block: int) -> float:
        """Theoretical omega for m multiplies per 2×2 block"""
        return np.log2(multiplies_per_block)

def analyze_algorithms():
    """Run the corrected analysis"""
    print("=" * 70)
    print("CORRECTED OMEGA ANALYSIS")
    print("=" * 70)
    print()
    
    analyzer = MatrixMultiplicationAnalyzer()
    
    # Test sizes (powers of 2 for clean recursion)
    sizes = [2, 4, 8, 16, 32, 64, 128, 256]
    
    # Analyze different algorithms
    algorithms = {
        'Naive O(n³)': analyzer.naive_multiply_count,
        'Strassen (7 multiplies)': analyzer.strassen_multiply_count,
        'Penrose Basic (5 multiplies)': analyzer.penrose_basic_multiply_count,
    }
    
    results = {}
    
    print("OPERATION COUNTS:")
    print("-" * 60)
    print(f"{'n':<6} {'Naive':<12} {'Strassen':<12} {'Penrose-5':<12}")
    print("-" * 60)
    
    for n in sizes:
        naive = analyzer.naive_multiply_count(n)
        strassen = analyzer.strassen_multiply_count(n)
        penrose = analyzer.penrose_basic_multiply_count(n)
        
        print(f"{n:<6} {naive:<12} {strassen:<12} {penrose:<12}")
    
    print()
    print("COMPUTED OMEGA VALUES:")
    print("-" * 60)
    
    # Compute omega for each algorithm
    for name, func in algorithms.items():
        counts = [func(n) for n in sizes]
        omega = analyzer.compute_omega(counts, sizes)
        theoretical = None
        
        if 'Strassen' in name:
            theoretical = np.log2(7)  # ≈ 2.807
        elif 'Penrose' in name and '5' in name:
            theoretical = np.log2(5)  # ≈ 2.322
            
        results[name] = {
            'counts': counts,
            'omega': omega,
            'theoretical': theoretical
        }
        
        print(f"{name:<30}: ω = {omega:.4f}", end="")
        if theoretical:
            print(f" (theoretical: {theoretical:.4f})")
        else:
            print()
    
    print()
    print("THEORETICAL ANALYSIS:")
    print("-" * 60)
    print("For recursive algorithm with m multiplies per 2×2 block:")
    print("T(n) = m * T(n/2)")
    print("Solution: T(n) = n^(log₂ m)")
    print()
    print("m = 8 (naive): ω = log₂(8) = 3.000")
    print("m = 7 (Strassen): ω = log₂(7) = 2.807")
    print("m = 5 (Penrose claim): ω = log₂(5) = 2.322")
    print("m = 4.73 (current record): ω = log₂(4.73) = 2.3728...")
    print()
    
    # What would we need to beat the record?
    target_omega = 2.3078
    required_m = 2 ** target_omega
    print(f"To achieve ω = {target_omega}, we need m ≤ {required_m:.3f} multiplies per 2×2 block")
    
    print()
    print("KEY INSIGHTS:")
    print("-" * 60)
    print("1. Parallelism doesn't change ω (it's a constant factor speedup)")
    print("2. Error correction only helps if it reduces m, not as a percentage")
    print("3. Our claimed 5-multiply algorithm gives ω ≈ 2.322")
    print("4. This is better than Strassen but NOT below 2.3078")
    print("5. We need ≤ 4.95 multiplies per 2×2 to break the record")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    
    for name, data in results.items():
        if data['omega']:
            plt.loglog(sizes, data['counts'], 'o-', linewidth=2, markersize=8,
                      label=f"{name} (ω={data['omega']:.3f})")
    
    # Reference lines
    n_range = np.array([2, 300])
    plt.loglog(n_range, n_range**2.3078, 'r--', alpha=0.5, 
               label='Current record (ω=2.3078)', linewidth=2)
    plt.loglog(n_range, n_range**2, 'g--', alpha=0.5, 
               label='Theoretical limit (ω=2.0)', linewidth=2)
    
    plt.xlabel('Matrix Size (n)', fontsize=12)
    plt.ylabel('Number of Multiplications', fontsize=12)
    plt.title('Corrected Operation Count Analysis', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('corrected_omega_analysis.png', dpi=150)
    print("\nPlot saved as corrected_omega_analysis.png")
    
    # Can we do better than 5?
    print("\n" + "=" * 70)
    print("WHAT WOULD IT TAKE TO BREAK ω = 2.3078?")
    print("=" * 70)
    print()
    print("We need a 2×2 multiplication algorithm using < 4.95 multiplies.")
    print()
    print("Possibilities:")
    print("1. Find a 4-multiply algorithm (ω = 2.0) - likely impossible")
    print("2. Use larger base case (3×3 with < 21 multiplies)")
    print("3. Exploit Penrose quantum error correction more deeply")
    print("4. Use non-uniform algorithms (different m at different scales)")
    print()
    print("The Penrose 5-multiply algorithm (if it works) gives us:")
    print(f"ω = log₂(5) = {np.log2(5):.4f}")
    print("This is close but not quite below 2.3078.")

if __name__ == "__main__":
    analyze_algorithms()
