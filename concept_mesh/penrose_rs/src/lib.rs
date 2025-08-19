//! Penrose Tensor Acceleration Engine - Full Implementation
//! High-performance Rust backend for fractal soliton memory operations
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

/// Global engine state
static ENGINE_INITIALIZED: AtomicBool = AtomicBool::new(false);
static ENGINE_CONFIG: Mutex<Option<EngineConfig>> = Mutex::new(None);

#[derive(Clone)]
struct EngineConfig {
    max_threads: usize,
    cache_size_mb: usize,
    enable_gpu: bool,
    precision: String,
}

/// Manual dot product for f32 slices
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// L2-norm of a vector
fn norm(v: &[f32]) -> f32 {
    dot(v, v).sqrt()
}

/// Initialize the Penrose engine with configuration
#[pyfunction]
fn initialize_engine(
    max_threads: usize,
    cache_size_mb: usize,
    enable_gpu: bool,
    precision: String,
) -> PyResult<PyObject> {
    let config = EngineConfig {
        max_threads,
        cache_size_mb,
        enable_gpu,
        precision: precision.clone(),
    };
    
    // Set rayon thread pool
    rayon::ThreadPoolBuilder::new()
        .num_threads(max_threads)
        .build_global()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Thread pool error: {}", e)))?;
    
    // Store config
    {
        let mut cfg = ENGINE_CONFIG.lock().unwrap();
        *cfg = Some(config);
    }
    
    ENGINE_INITIALIZED.store(true, Ordering::Relaxed);
    
    Python::with_gil(|py| {
        let result = PyDict::new(py);
        result.set_item("success", true)?;
        result.set_item("gpu_available", enable_gpu)?;
        result.set_item("thread_count", max_threads)?;
        result.set_item("precision", precision)?;
        Ok(result.into())
    })
}

/// Get engine information and status
#[pyfunction]
fn get_engine_info() -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let result = PyDict::new(py);
        
        if ENGINE_INITIALIZED.load(Ordering::Relaxed) {
            let config_lock = ENGINE_CONFIG.lock().unwrap();
            if let Some(ref config) = *config_lock {
                result.set_item("version", "0.1.1")?;
                result.set_item("gpu_enabled", config.enable_gpu)?;
                result.set_item("thread_count", config.max_threads)?;
                result.set_item("cache_size_mb", config.cache_size_mb)?;
                result.set_item("precision", &config.precision)?;
                result.set_item("initialized", true)?;
            } else {
                result.set_item("initialized", false)?;
                result.set_item("error", "Config not found")?;
            }
        } else {
            result.set_item("initialized", false)?;
            result.set_item("error", "Engine not initialized")?;
        }
        
        Ok(result.into())
    })
}

/// Evolve lattice field using phase-curvature coupling
#[pyfunction]
fn evolve_lattice_field(
    lattice: Vec<Vec<f32>>,
    phase_field: Vec<Vec<f32>>,
    curvature_field: Vec<Vec<f32>>,
    dt: f32,
) -> PyResult<Vec<Vec<f32>>> {
    if !ENGINE_INITIALIZED.load(Ordering::Relaxed) {
        return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "Engine not initialized"
        ));
    }
    
    let height = lattice.len();
    if height == 0 {
        return Ok(lattice);
    }
    let width = lattice[0].len();
    
    // Validate dimensions
    if phase_field.len() != height || curvature_field.len() != height {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Field dimensions must match lattice"
        ));
    }
    
    // Parallel evolution using rayon
    let evolved: Vec<Vec<f32>> = (0..height).into_par_iter().map(|i| {
        (0..width).map(|j| {
            // Current values
            let current = lattice[i][j];
            let phase = phase_field[i][j];
            let curvature = curvature_field[i][j];
            
            // Compute gradients (simple finite differences)
            let phase_grad_x = if j < width - 1 { phase_field[i][j + 1] - phase } else { 0.0 };
            let phase_grad_y = if i < height - 1 { phase_field[i + 1][j] - phase } else { 0.0 };
            
            // Phase-curvature coupling evolution
            let phase_energy = 0.1 * (phase_grad_x * phase_grad_x + phase_grad_y * phase_grad_y);
            let curvature_coupling = 0.05 * curvature * phase.cos();
            let evolution_term = phase_energy + curvature_coupling;
            
            current + dt * evolution_term
        }).collect()
    }).collect();
    
    Ok(evolved)
}

/// Compute quantum phase entanglement between solitons
#[pyfunction]
fn compute_phase_entanglement(
    soliton_positions: Vec<Vec<f32>>,
    phases: Vec<f32>,
    coupling_strength: f32,
) -> PyResult<Vec<Vec<f32>>> {
    if !ENGINE_INITIALIZED.load(Ordering::Relaxed) {
        return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "Engine not initialized"
        ));
    }
    
    let n_solitons = soliton_positions.len();
    if phases.len() != n_solitons {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Phases array must match number of solitons"
        ));
    }
    
    // Parallel computation of entanglement matrix
    let entanglement: Vec<Vec<f32>> = (0..n_solitons).into_par_iter().map(|i| {
        (0..n_solitons).map(|j| {
            if i == j {
                0.0 // No self-entanglement
            } else {
                // Distance between solitons
                let pos_i = &soliton_positions[i];
                let pos_j = &soliton_positions[j];
                let distance = ((pos_i[0] - pos_j[0]).powi(2) + (pos_i[1] - pos_j[1]).powi(2)).sqrt();
                
                // Phase difference
                let phase_diff = (phases[i] - phases[j]).abs();
                
                // Entanglement strength with distance decay
                coupling_strength * (-distance / 10.0).exp() * phase_diff.cos()
            }
        }).collect()
    }).collect();
    
    Ok(entanglement)
}

/// Encode curvature field to phase/amplitude representation
#[pyfunction]
fn curvature_to_phase_encode(
    curvature_field: Vec<Vec<f32>>,
    encoding_mode: String,
) -> PyResult<PyObject> {
    if !ENGINE_INITIALIZED.load(Ordering::Relaxed) {
        return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "Engine not initialized"
        ));
    }
    
    let height = curvature_field.len();
    if height == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Empty curvature field"
        ));
    }
    let width = curvature_field[0].len();
    
    // Parallel encoding based on mode
    let (phase_field, amplitude_field): (Vec<Vec<f32>>, Vec<Vec<f32>>) = match encoding_mode.as_str() {
        "log_phase" => {
            let data: Vec<(Vec<f32>, Vec<f32>)> = (0..height).into_par_iter().map(|i| {
                let phase_row: Vec<f32> = curvature_field[i].iter().map(|&k| {
                    let safe_k = k.max(1e-8);
                    (safe_k.ln() % (2.0 * std::f32::consts::PI)) - std::f32::consts::PI
                }).collect();
                
                let amplitude_row: Vec<f32> = curvature_field[i].iter().map(|&k| {
                    let safe_k = k.max(1e-8);
                    1.0 / (1.0 + safe_k * safe_k)
                }).collect();
                
                (phase_row, amplitude_row)
            }).collect();
            
            let phases: Vec<Vec<f32>> = data.iter().map(|(p, _)| p.clone()).collect();
            let amplitudes: Vec<Vec<f32>> = data.iter().map(|(_, a)| a.clone()).collect();
            (phases, amplitudes)
        },
        "tanh" => {
            let data: Vec<(Vec<f32>, Vec<f32>)> = (0..height).into_par_iter().map(|i| {
                let phase_row: Vec<f32> = curvature_field[i].iter().map(|&k| {
                    k.tanh() * std::f32::consts::PI
                }).collect();
                
                let amplitude_row: Vec<f32> = curvature_field[i].iter().map(|&k| {
                    1.0 / k.cosh()
                }).collect();
                
                (phase_row, amplitude_row)
            }).collect();
            
            let phases: Vec<Vec<f32>> = data.iter().map(|(p, _)| p.clone()).collect();
            let amplitudes: Vec<Vec<f32>> = data.iter().map(|(_, a)| a.clone()).collect();
            (phases, amplitudes)
        },
        _ => { // "direct"
            let phases: Vec<Vec<f32>> = curvature_field.iter().map(|row| {
                row.iter().map(|&k| (k % (2.0 * std::f32::consts::PI)) - std::f32::consts::PI).collect()
            }).collect();
            
            let amplitudes: Vec<Vec<f32>> = (0..height).map(|i| {
                vec![1.0; width]
            }).collect();
            
            (phases, amplitudes)
        }
    };
    
    Python::with_gil(|py| {
        let result = PyDict::new(py);
        result.set_item("phase", phase_field)?;
        result.set_item("amplitude", amplitude_field)?;
        Ok(result.into())
    })
}

/// Shutdown the engine and clean up resources
#[pyfunction]
fn shutdown_engine() -> PyResult<()> {
    ENGINE_INITIALIZED.store(false, Ordering::Relaxed);
    
    // Clear config
    {
        let mut cfg = ENGINE_CONFIG.lock().unwrap();
        *cfg = None;
    }
    
    Ok(())
}

/// Original similarity functions (kept for backward compatibility)
#[pyfunction]
fn compute_similarity(v1: Vec<f32>, v2: Vec<f32>) -> PyResult<f32> {
    if v1.len() != v2.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Vectors must have equal length",
        ));
    }
    let sim = dot(&v1, &v2) / (norm(&v1) * norm(&v2));
    Ok(sim)
}

#[pyfunction]
fn batch_similarity(query: Vec<f32>, corpus: Vec<Vec<f32>>) -> PyResult<Vec<f32>> {
    let qn = norm(&query);
    let sims: Vec<f32> = corpus
        .par_iter()
        .map(|row| dot(&query, row) / (qn * norm(row)))
        .collect();
    Ok(sims)
}

/// Python module definition
#[pymodule]
fn penrose_engine_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    // Core Penrose engine functions
    m.add_function(wrap_pyfunction!(initialize_engine, m)?)?;
    m.add_function(wrap_pyfunction!(get_engine_info, m)?)?;
    m.add_function(wrap_pyfunction!(evolve_lattice_field, m)?)?;
    m.add_function(wrap_pyfunction!(compute_phase_entanglement, m)?)?;
    m.add_function(wrap_pyfunction!(curvature_to_phase_encode, m)?)?;
    m.add_function(wrap_pyfunction!(shutdown_engine, m)?)?;
    
    // Backward compatibility functions
    m.add_function(wrap_pyfunction!(compute_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(batch_similarity, m)?)?;
    
    Ok(())
}
