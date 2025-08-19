use ndarray::{Array2, ArrayView2};
use num_complex::Complex64;
use std::collections::HashMap;

/// Trait for generating χ-parametrized Laplacian matrices
pub trait ChiLaplacian {
    /// Generate Laplacian with chi modification
    fn with_chi(&self, chi: f64) -> Array2<Complex64>;
    
    /// Get the base Laplacian (chi = 0)
    fn base_laplacian(&self) -> &Array2<Complex64>;
    
    /// Apply chi-dependent self-energy correction
    fn apply_self_energy(&self, laplacian: &mut Array2<Complex64>, chi: f64, poles: &[(f64, f64)]);
}

/// Fast χ-parametrized Laplacian generator
pub struct ChiLaplacianGenerator {
    base_laplacian: Array2<Complex64>,
    n_sites: usize,
    gamma_mv: f64,
    gap: f64,
}

impl ChiLaplacianGenerator {
    /// Create a new chi Laplacian generator
    pub fn new(n_sites: usize, gamma_mv: f64, gap: f64) -> Self {
        // Build base nearest-neighbor Laplacian
        let mut base = Array2::<Complex64>::zeros((n_sites, n_sites));
        
        // Diagonal: -2
        for i in 0..n_sites {
            base[[i, i]] = Complex64::new(-2.0, 0.0);
        }
        
        // Off-diagonal: +1 for nearest neighbors
        for i in 0..n_sites - 1 {
            base[[i, i + 1]] = Complex64::new(1.0, 0.0);
            base[[i + 1, i]] = Complex64::new(1.0, 0.0);
        }
        
        // Periodic boundary conditions
        base[[0, n_sites - 1]] = Complex64::new(1.0, 0.0);
        base[[n_sites - 1, 0]] = Complex64::new(1.0, 0.0);
        
        Self {
            base_laplacian: base,
            n_sites,
            gamma_mv,
            gap,
        }
    }
    
    /// Build Laplacian from custom coupling matrix
    pub fn from_coupling_matrix(coupling: ArrayView2<f64>, gamma_mv: f64, gap: f64) -> Self {
        let n_sites = coupling.shape()[0];
        let mut base = Array2::<Complex64>::zeros((n_sites, n_sites));
        
        // Convert coupling to complex Laplacian
        for i in 0..n_sites {
            let mut row_sum = 0.0;
            for j in 0..n_sites {
                if i != j {
                    let coupling_ij = coupling[[i, j]];
                    base[[i, j]] = Complex64::new(coupling_ij, 0.0);
                    row_sum += coupling_ij;
                }
            }
            // Diagonal is negative sum of row (for conservation)
            base[[i, i]] = Complex64::new(-row_sum, 0.0);
        }
        
        Self {
            base_laplacian: base,
            n_sites,
            gamma_mv,
            gap,
        }
    }
}

impl ChiLaplacian for ChiLaplacianGenerator {
    fn with_chi(&self, chi: f64) -> Array2<Complex64> {
        // Start with base Laplacian
        let mut laplacian = self.base_laplacian.clone();
        
        // Add chi-dependent corrections
        if chi.abs() > 1e-10 {
            // Diagonal self-energy contribution
            let diagonal_correction = -Complex64::new(0.0, self.gamma_mv * chi * self.gap);
            for i in 0..self.n_sites {
                laplacian[[i, i]] += diagonal_correction;
            }
            
            // Off-diagonal chi coupling
            let off_diag = -self.gamma_mv * chi * self.gap / 2.0;
            for i in 0..self.n_sites - 1 {
                laplacian[[i, i + 1]] += Complex64::new(off_diag, 0.0);
                laplacian[[i + 1, i]] += Complex64::new(off_diag, 0.0);
            }
        }
        
        laplacian
    }
    
    fn base_laplacian(&self) -> &Array2<Complex64> {
        &self.base_laplacian
    }
    
    fn apply_self_energy(&self, laplacian: &mut Array2<Complex64>, chi: f64, poles: &[(f64, f64)]) {
        // Apply continuous self-energy (already done in with_chi)
        
        // Add pole contributions
        for (omega_pole, gamma_pole) in poles {
            // Simplified pole contribution - localized at center
            let center = self.n_sites / 2;
            let width = (self.n_sites as f64 / 20.0).max(1.0) as usize;
            
            for i in center.saturating_sub(width)..=(center + width).min(self.n_sites - 1) {
                let weight = ((i as f64 - center as f64).powi(2) / (2.0 * width as f64).powi(2)).exp();
                let pole_contrib = Complex64::new(*gamma_pole * weight, 0.0);
                laplacian[[i, i]] += pole_contrib;
            }
        }
    }
}

/// Cache for pre-computed chi Laplacians
pub struct ChiLaplacianCache {
    generator: ChiLaplacianGenerator,
    cache: HashMap<u64, Array2<Complex64>>,
    cache_size: usize,
}

impl ChiLaplacianCache {
    pub fn new(generator: ChiLaplacianGenerator, cache_size: usize) -> Self {
        Self {
            generator,
            cache: HashMap::with_capacity(cache_size),
            cache_size,
        }
    }
    
    /// Get Laplacian for given chi value (with caching)
    pub fn get(&mut self, chi: f64) -> &Array2<Complex64> {
        // Quantize chi to improve cache hits
        let chi_key = (chi * 1000.0).round() as u64;
        
        // Check cache
        if self.cache.contains_key(&chi_key) {
            return &self.cache[&chi_key];
        }
        
        // Compute if not cached
        let laplacian = self.generator.with_chi(chi);
        
        // Evict oldest if cache full
        if self.cache.len() >= self.cache_size {
            // Simple FIFO eviction
            let oldest_key = *self.cache.keys().next().unwrap();
            self.cache.remove(&oldest_key);
        }
        
        self.cache.insert(chi_key, laplacian);
        &self.cache[&chi_key]
    }
    
    /// Clear the cache
    pub fn clear(&mut self) {
        self.cache.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_chi_laplacian_generator() {
        let gen = ChiLaplacianGenerator::new(10, 1.0, 1.0);
        
        // Test chi = 0 gives base Laplacian
        let l0 = gen.with_chi(0.0);
        assert_relative_eq!(l0[[0, 0]].re, -2.0, epsilon = 1e-10);
        assert_relative_eq!(l0[[0, 1]].re, 1.0, epsilon = 1e-10);
        
        // Test chi != 0 modifies the Laplacian
        let l1 = gen.with_chi(0.5);
        assert_ne!(l1[[0, 0]], l0[[0, 0]]);
    }
    
    #[test]
    fn test_laplacian_cache() {
        let gen = ChiLaplacianGenerator::new(10, 1.0, 1.0);
        let mut cache = ChiLaplacianCache::new(gen, 5);
        
        // First access computes
        let _l1 = cache.get(0.5);
        
        // Second access should use cache
        let _l2 = cache.get(0.5);
        
        // Different chi computes new
        let _l3 = cache.get(0.7);
        
        assert_eq!(cache.cache.len(), 2);
    }
}
