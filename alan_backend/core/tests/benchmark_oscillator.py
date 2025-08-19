# Copyright 2025 ALAN Team and contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Patent Peace / Retaliation Notice:
#   As stated in Section 3 of the Apache 2.0 License, any entity that
#   initiates patent litigation (including a cross-claim or counterclaim)
#   alleging that this software or a contribution embodied within it
#   infringes a patent shall have all patent licenses granted herein
#   terminated as of the date such litigation is filed.

"""
Benchmarks for the Banksy oscillator components.

These benchmarks measure the performance of the oscillator substrate
with varying numbers of oscillators and coupling configurations.
"""

import pytest
import numpy as np
from alan_backend.core.oscillator.banksy_oscillator import BanksyOscillator, BanksyConfig, SpinVector


class TestBanksyOscillatorBenchmarks:
    """Benchmarks for the Banksy oscillator."""
    
    @pytest.mark.parametrize("n_oscillators", [16, 32, 64, 128])
    def test_oscillator_step(self, benchmark, n_oscillators):
        """Benchmark the step method with different numbers of oscillators."""
        # Create oscillator with default parameters
        config = BanksyConfig(gamma=0.1, epsilon=0.01, eta_damp=1e-4, dt=0.01)
        oscillator = BanksyOscillator(n_oscillators, config)
        
        # Default all-to-all coupling
        
        # Run benchmark
        benchmark.group = "Oscillator Step"
        benchmark.name = f"{n_oscillators} oscillators"
        benchmark(oscillator.step)
    
    @pytest.mark.parametrize("coupling_density", [0.1, 0.5, 1.0])
    def test_coupling_density(self, benchmark, coupling_density):
        """Benchmark the effect of coupling density on performance."""
        n_oscillators = 64
        
        # Create oscillator with default parameters
        config = BanksyConfig(gamma=0.1, epsilon=0.01, eta_damp=1e-4, dt=0.01)
        oscillator = BanksyOscillator(n_oscillators, config)
        
        # Create coupling matrix with specific density
        np.random.seed(42)  # For reproducibility
        coupling = np.random.uniform(0, 0.2, (n_oscillators, n_oscillators))
        
        # Set some connections to zero based on density
        mask = np.random.random((n_oscillators, n_oscillators)) > coupling_density
        coupling[mask] = 0.0
        np.fill_diagonal(coupling, 0.0)  # No self-coupling
        
        oscillator.set_coupling(coupling)
        
        # Run benchmark
        benchmark.group = "Coupling Density"
        benchmark.name = f"{coupling_density:.1f} density"
        benchmark(oscillator.step)
    
    @pytest.mark.parametrize("spin_substeps", [1, 4, 8, 16])
    def test_spin_substeps(self, benchmark, spin_substeps):
        """Benchmark the effect of spin substeps on performance."""
        n_oscillators = 32
        
        # Create oscillator
        config = BanksyConfig(gamma=0.1, epsilon=0.01, eta_damp=1e-4, dt=0.01)
        oscillator = BanksyOscillator(n_oscillators, config)
        
        # Create partial function with specific substeps
        def step_with_substeps():
            oscillator.step(spin_substeps=spin_substeps)
        
        # Run benchmark
        benchmark.group = "Spin Substeps"
        benchmark.name = f"{spin_substeps} substeps"
        benchmark(step_with_substeps)
    
    def test_synchronization_speed(self, benchmark):
        """Benchmark how quickly the oscillator network can synchronize."""
        n_oscillators = 32
        
        # Create oscillator
        config = BanksyConfig(gamma=0.1, epsilon=0.01, eta_damp=1e-4, dt=0.01)
        oscillator = BanksyOscillator(n_oscillators, config)
        
        # Strong all-to-all coupling for fast synchronization
        coupling = np.ones((n_oscillators, n_oscillators)) * 0.2
        np.fill_diagonal(coupling, 0.0)
        oscillator.set_coupling(coupling)
        
        # Random initial phases
        np.random.seed(42)
        oscillator.phases = np.random.uniform(0, 2*np.pi, n_oscillators)
        
        # Function to run until synchronization
        def run_until_sync(threshold=0.95, max_steps=1000):
            for _ in range(max_steps):
                oscillator.step()
                r = oscillator.order_parameter()
                if r > threshold:
                    break
            return oscillator.order_parameter()
        
        # Run benchmark
        benchmark.group = "Synchronization"
        benchmark.name = "Time to sync r>0.95"
        result = benchmark(run_until_sync)
        
        # Verify it actually synchronized
        assert result > 0.95, f"Failed to synchronize, reached r={result}"


class TestSpinVectorBenchmarks:
    """Benchmarks for the SpinVector component."""
    
    def test_spin_vector_creation(self, benchmark):
        """Benchmark creating many spin vectors."""
        def create_vectors(n=1000):
            vectors = []
            for _ in range(n):
                x = np.random.uniform(-1, 1)
                y = np.random.uniform(-1, 1)
                z = np.random.uniform(-1, 1)
                vectors.append(SpinVector(x, y, z))
            return vectors
        
        benchmark.group = "SpinVector"
        benchmark.name = "Creation"
        vectors = benchmark(create_vectors)
        assert len(vectors) == 1000
    
    def test_spin_vector_dot_product(self, benchmark):
        """Benchmark computing dot products between spin vectors."""
        # Create vectors first
        np.random.seed(42)
        n = 1000
        vectors = []
        for _ in range(n):
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)
            z = np.random.uniform(-1, 1)
            vectors.append(SpinVector(x, y, z))
        
        # Function to compute all pairwise dot products
        def compute_dots(vectors, n_pairs=1000):
            results = []
            for _ in range(n_pairs):
                i = np.random.randint(0, len(vectors))
                j = np.random.randint(0, len(vectors))
                results.append(vectors[i].dot(vectors[j]))
            return results
        
        benchmark.group = "SpinVector"
        benchmark.name = "Dot Product"
        dots = benchmark(compute_dots, vectors)
        assert len(dots) == 1000


if __name__ == "__main__":
    # This allows running the benchmarks directly
    pytest.main(["-xvs", __file__])
