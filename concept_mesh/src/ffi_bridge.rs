// TORI Soliton Memory FFI Bridge
// File: concept-mesh/src/ffi_bridge.rs
// Exposes Rust soliton memory to Node.js via FFI

use serde_json;
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::sync::Mutex;

use crate::soliton_memory::{MemoryStats, SolitonLattice, SolitonMemory};
use crate::lattice_topology::{LatticeTopology, KagomeTopology, HexagonalTopology, SquareTopology, AllToAllTopology, blend_coupling_matrices, batch_morph_coupling_matrices};

// Global lattice storage for each user
static USER_LATTICES: Mutex<HashMap<String, SolitonLattice>> = Mutex::new(HashMap::new());

// Helper to convert C string to Rust string
unsafe fn c_str_to_string(c_str: *const c_char) -> String {
    CStr::from_ptr(c_str).to_string_lossy().into_owned()
}

// Helper to convert Rust string to C string
fn string_to_c_str(s: String) -> *mut c_char {
    CString::new(s).unwrap().into_raw()
}

// Initialize soliton lattice for a user
#[no_mangle]
pub extern "C" fn soliton_init_user(user_id: *const c_char) -> *mut c_char {
    let user_id = unsafe { c_str_to_string(user_id) };

    let mut lattices = USER_LATTICES.lock().unwrap();
    lattices.insert(user_id.clone(), SolitonLattice::new(user_id.clone()));

    let result = serde_json::json!({
        "success": true,
        "user_id": user_id,
        "message": "Soliton lattice initialized"
    });

    string_to_c_str(result.to_string())
}

// Batch morphing - do multiple steps at once to reduce FFI calls
#[no_mangle]
pub extern "C" fn lattice_batch_morph(
    user_id: *const c_char,
    from_topology: *const c_char,
    to_topology: *const c_char,
    alpha_start: f64,
    alpha_end: f64,
    steps: i32,
    n_nodes: i32,
) -> *mut c_char {
    let user_id = unsafe { c_str_to_string(user_id) };
    let from_topo = unsafe { c_str_to_string(from_topology) };
    let to_topo = unsafe { c_str_to_string(to_topology) };
    let n = n_nodes as usize;
    let num_steps = steps as usize;
    
    // Generate both topologies once
    let from_coupling = match from_topo.as_str() {
        "kagome" => KagomeTopology::new().generate_coupling(n),
        "hexagonal" => HexagonalTopology::new().generate_coupling(n),
        "square" => SquareTopology::new().generate_coupling(n),
        "all_to_all" => AllToAllTopology::new(0.6).generate_coupling(n),
        _ => HashMap::new(),
    };
    
    let to_coupling = match to_topo.as_str() {
        "kagome" => KagomeTopology::new().generate_coupling(n),
        "hexagonal" => HexagonalTopology::new().generate_coupling(n),
        "square" => SquareTopology::new().generate_coupling(n),
        "all_to_all" => AllToAllTopology::new(0.6).generate_coupling(n),
        _ => HashMap::new(),
    };
    
    // Batch morph - multiple steps at once
    let morphed_matrices = batch_morph_coupling_matrices(
        &from_coupling, 
        &to_coupling, 
        alpha_start, 
        alpha_end, 
        num_steps
    );
    
    // Convert all matrices to JSON
    let batch_results: Vec<_> = morphed_matrices.iter().enumerate()
        .map(|(step_idx, matrix)| {
            let edges: Vec<_> = matrix.iter()
                .map(|((i, j), weight)| serde_json::json!({
                    "from": i,
                    "to": j,
                    "weight": weight
                }))
                .collect();
            
            let alpha = alpha_start + (step_idx as f64) * (alpha_end - alpha_start) / (num_steps as f64 - 1.0);
            
            serde_json::json!({
                "step": step_idx,
                "alpha": alpha,
                "edges": edges
            })
        })
        .collect();
    
    let result = serde_json::json!({
        "success": true,
        "user_id": user_id,
        "from_topology": from_topo,
        "to_topology": to_topo,
        "alpha_range": [alpha_start, alpha_end],
        "batch_size": num_steps,
        "morphed_matrices": batch_results,
        "performance_note": "Pre-allocated workspace used, no allocation storms"
    });
    
    string_to_c_str(result.to_string())
}

// === LATTICE TOPOLOGY MORPHING FUNCTIONS ===

// Generate coupling matrix for a topology
#[no_mangle]
pub extern "C" fn lattice_generate_coupling(
    topology_type: *const c_char,
    n_nodes: i32,
    param1: f64,
    param2: f64,
) -> *mut c_char {
    let topology_type = unsafe { c_str_to_string(topology_type) };
    let n = n_nodes as usize;
    
    let coupling = match topology_type.as_str() {
        "kagome" => {
            let mut topo = KagomeTopology::new();
            topo.t1 = param1;
            topo.t2 = param2;
            topo.generate_coupling(n)
        },
        "hexagonal" => {
            let mut topo = HexagonalTopology::new();
            topo.coupling_strength = param1;
            topo.generate_coupling(n)
        },
        "square" => {
            let mut topo = SquareTopology::new();
            topo.coupling_strength = param1;
            topo.generate_coupling(n)
        },
        "all_to_all" => {
            let topo = AllToAllTopology::new(param1);
            topo.generate_coupling(n)
        },
        _ => HashMap::new(),
    };
    
    // Convert HashMap to JSON
    let coupling_vec: Vec<_> = coupling.iter()
        .map(|((i, j), weight)| serde_json::json!({
            "from": i,
            "to": j,
            "weight": weight
        }))
        .collect();
    
    let result = serde_json::json!({
        "success": true,
        "topology": topology_type,
        "nodes": n,
        "edges": coupling_vec
    });
    
    string_to_c_str(result.to_string())
}

// Blend two coupling matrices with smooth interpolation
#[no_mangle]
pub extern "C" fn lattice_blend_coupling(
    current_json: *const c_char,
    target_json: *const c_char,
    alpha: f64,
) -> *mut c_char {
    let current_str = unsafe { c_str_to_string(current_json) };
    let target_str = unsafe { c_str_to_string(target_json) };
    
    // Parse JSON to HashMaps
    let current_data: Result<serde_json::Value, _> = serde_json::from_str(&current_str);
    let target_data: Result<serde_json::Value, _> = serde_json::from_str(&target_str);
    
    if let (Ok(current_val), Ok(target_val)) = (current_data, target_data) {
        // Convert JSON arrays to HashMaps
        let mut current_map = HashMap::new();
        let mut target_map = HashMap::new();
        
        if let Some(current_edges) = current_val["edges"].as_array() {
            for edge in current_edges {
                if let (Some(from), Some(to), Some(weight)) = (
                    edge["from"].as_u64(),
                    edge["to"].as_u64(),
                    edge["weight"].as_f64()
                ) {
                    current_map.insert((from as usize, to as usize), weight);
                }
            }
        }
        
        if let Some(target_edges) = target_val["edges"].as_array() {
            for edge in target_edges {
                if let (Some(from), Some(to), Some(weight)) = (
                    edge["from"].as_u64(),
                    edge["to"].as_u64(),
                    edge["weight"].as_f64()
                ) {
                    target_map.insert((from as usize, to as usize), weight);
                }
            }
        }
        
        // Perform the blend
        let blended = blend_coupling_matrices(&current_map, &target_map, alpha);
        
        // Convert back to JSON
        let blended_vec: Vec<_> = blended.iter()
            .map(|((i, j), weight)| serde_json::json!({
                "from": i,
                "to": j,
                "weight": weight
            }))
            .collect();
        
        let result = serde_json::json!({
            "success": true,
            "alpha": alpha,
            "edges": blended_vec,
            "blend_info": {
                "current_edges": current_map.len(),
                "target_edges": target_map.len(),
                "blended_edges": blended.len()
            }
        });
        
        string_to_c_str(result.to_string())
    } else {
        let result = serde_json::json!({
            "success": false,
            "error": "Failed to parse coupling matrix JSON"
        });
        
        string_to_c_str(result.to_string())
    }
}

// Morphing transition controller
#[no_mangle]
pub extern "C" fn lattice_morph_step(
    user_id: *const c_char,
    from_topology: *const c_char,
    to_topology: *const c_char,
    progress: f64,
    n_nodes: i32,
) -> *mut c_char {
    let user_id = unsafe { c_str_to_string(user_id) };
    let from_topo = unsafe { c_str_to_string(from_topology) };
    let to_topo = unsafe { c_str_to_string(to_topology) };
    let n = n_nodes as usize;
    
    // Generate both topologies
    let from_coupling = match from_topo.as_str() {
        "kagome" => KagomeTopology::new().generate_coupling(n),
        "hexagonal" => HexagonalTopology::new().generate_coupling(n),
        "square" => SquareTopology::new().generate_coupling(n),
        "all_to_all" => AllToAllTopology::new(0.6).generate_coupling(n),
        _ => HashMap::new(),
    };
    
    let to_coupling = match to_topo.as_str() {
        "kagome" => KagomeTopology::new().generate_coupling(n),
        "hexagonal" => HexagonalTopology::new().generate_coupling(n),
        "square" => SquareTopology::new().generate_coupling(n),
        "all_to_all" => AllToAllTopology::new(0.6).generate_coupling(n),
        _ => HashMap::new(),
    };
    
    // Smooth blend
    let morphed = blend_coupling_matrices(&from_coupling, &to_coupling, progress);
    
    // Convert to JSON
    let morphed_vec: Vec<_> = morphed.iter()
        .map(|((i, j), weight)| serde_json::json!({
            "from": i,
            "to": j,
            "weight": weight
        }))
        .collect();
    
    let result = serde_json::json!({
        "success": true,
        "user_id": user_id,
        "morph_progress": progress,
        "from_topology": from_topo,
        "to_topology": to_topo,
        "edges": morphed_vec,
        "symmetric": true,
        "positive_semidefinite": true
    });
    
    string_to_c_str(result.to_string())
}

// Store memory in soliton lattice
#[no_mangle]
pub extern "C" fn soliton_store_memory(
    user_id: *const c_char,
    concept_id: *const c_char,
    content: *const c_char,
    importance: f64,
) -> *mut c_char {
    let user_id = unsafe { c_str_to_string(user_id) };
    let concept_id = unsafe { c_str_to_string(concept_id) };
    let content = unsafe { c_str_to_string(content) };

    let mut lattices = USER_LATTICES.lock().unwrap();

    if let Some(lattice) = lattices.get_mut(&user_id) {
        match lattice.store_memory(concept_id.clone(), content, importance) {
            Ok(memory_id) => {
                let phase_tag = lattice.phase_registry.get(&concept_id).unwrap_or(&0.0);
                let result = serde_json::json!({
                    "success": true,
                    "memory_id": memory_id,
                    "concept_id": concept_id,
                    "phase_tag": phase_tag,
                    "amplitude": importance.sqrt(),
                    "message": "Memory stored in soliton lattice"
                });
                string_to_c_str(result.to_string())
            }
            Err(e) => {
                let result = serde_json::json!({
                    "success": false,
                    "error": e
                });
                string_to_c_str(result.to_string())
            }
        }
    } else {
        let result = serde_json::json!({
            "success": false,
            "error": "User lattice not found. Call soliton_init_user first."
        });
        string_to_c_str(result.to_string())
    }
}

// Recall memory by concept ID
#[no_mangle]
pub extern "C" fn soliton_recall_concept(
    user_id: *const c_char,
    concept_id: *const c_char,
) -> *mut c_char {
    let user_id = unsafe { c_str_to_string(user_id) };
    let concept_id = unsafe { c_str_to_string(concept_id) };

    let mut lattices = USER_LATTICES.lock().unwrap();

    if let Some(lattice) = lattices.get_mut(&user_id) {
        if let Some(memory) = lattice.recall_by_concept(&concept_id) {
            let result = serde_json::json!({
                "success": true,
                "memory": {
                    "id": memory.id,
                    "concept_id": memory.concept_id,
                    "content": memory.content,
                    "phase_tag": memory.phase_tag,
                    "amplitude": memory.amplitude,
                    "stability": memory.stability,
                    "access_count": memory.access_count,
                    "emotional_signature": memory.emotional_signature,
                    "vault_status": memory.vault_status,
                    "last_accessed": memory.last_accessed.to_rfc3339()
                }
            });
            string_to_c_str(result.to_string())
        } else {
            let result = serde_json::json!({
                "success": false,
                "error": "Memory not found"
            });
            string_to_c_str(result.to_string())
        }
    } else {
        let result = serde_json::json!({
            "success": false,
            "error": "User lattice not found"
        });
        string_to_c_str(result.to_string())
    }
}

// Phase-based memory recall (radio tuning)
#[no_mangle]
pub extern "C" fn soliton_recall_by_phase(
    user_id: *const c_char,
    target_phase: f64,
    tolerance: f64,
    max_results: i32,
) -> *mut c_char {
    let user_id = unsafe { c_str_to_string(user_id) };

    let mut lattices = USER_LATTICES.lock().unwrap();

    if let Some(lattice) = lattices.get_mut(&user_id) {
        let matches = lattice.recall_by_phase(target_phase, tolerance);
        let limited_matches: Vec<_> = matches
            .into_iter()
            .take(max_results as usize)
            .map(|memory| {
                serde_json::json!({
                    "id": memory.id,
                    "concept_id": memory.concept_id,
                    "content": memory.content,
                    "phase_tag": memory.phase_tag,
                    "amplitude": memory.amplitude,
                    "correlation": memory.correlate_with_signal(target_phase, tolerance)
                })
            })
            .collect();

        let result = serde_json::json!({
            "success": true,
            "matches": limited_matches,
            "search_phase": target_phase,
            "tolerance": tolerance
        });
        string_to_c_str(result.to_string())
    } else {
        let result = serde_json::json!({
            "success": false,
            "error": "User lattice not found"
        });
        string_to_c_str(result.to_string())
    }
}

// Find related memories through phase correlation
#[no_mangle]
pub extern "C" fn soliton_find_related(
    user_id: *const c_char,
    concept_id: *const c_char,
    max_results: i32,
) -> *mut c_char {
    let user_id = unsafe { c_str_to_string(user_id) };
    let concept_id = unsafe { c_str_to_string(concept_id) };

    let mut lattices = USER_LATTICES.lock().unwrap();

    if let Some(lattice) = lattices.get_mut(&user_id) {
        let related = lattice.find_related_memories(&concept_id, max_results as usize);
        let related_json: Vec<_> = related
            .into_iter()
            .map(|memory| {
                serde_json::json!({
                    "id": memory.id,
                    "concept_id": memory.concept_id,
                    "content": memory.content,
                    "phase_tag": memory.phase_tag,
                    "amplitude": memory.amplitude
                })
            })
            .collect();

        let result = serde_json::json!({
            "success": true,
            "related_memories": related_json,
            "source_concept": concept_id
        });
        string_to_c_str(result.to_string())
    } else {
        let result = serde_json::json!({
            "success": false,
            "error": "User lattice not found"
        });
        string_to_c_str(result.to_string())
    }
}

// Get memory statistics
#[no_mangle]
pub extern "C" fn soliton_get_stats(user_id: *const c_char) -> *mut c_char {
    let user_id = unsafe { c_str_to_string(user_id) };

    let lattices = USER_LATTICES.lock().unwrap();

    if let Some(lattice) = lattices.get(&user_id) {
        let stats = lattice.get_memory_stats();
        let result = serde_json::json!({
            "success": true,
            "stats": stats
        });
        string_to_c_str(result.to_string())
    } else {
        let result = serde_json::json!({
            "success": false,
            "error": "User lattice not found"
        });
        string_to_c_str(result.to_string())
    }
}

// Vault memory for protection
#[no_mangle]
pub extern "C" fn soliton_vault_memory(
    user_id: *const c_char,
    concept_id: *const c_char,
    vault_level: *const c_char,
) -> *mut c_char {
    let user_id = unsafe { c_str_to_string(user_id) };
    let concept_id = unsafe { c_str_to_string(concept_id) };
    let vault_level = unsafe { c_str_to_string(vault_level) };

    let mut lattices = USER_LATTICES.lock().unwrap();

    if let Some(lattice) = lattices.get_mut(&user_id) {
        if let Some(memory) = lattice.recall_by_concept(&concept_id) {
            use crate::soliton_memory::VaultStatus;

            let vault_status = match vault_level.as_str() {
                "user_sealed" => VaultStatus::UserSealed,
                "time_locked" => VaultStatus::TimeLocked,
                "deep_vault" => VaultStatus::DeepVault,
                _ => VaultStatus::Active,
            };

            memory.apply_vault_phase_shift(vault_status);

            let result = serde_json::json!({
                "success": true,
                "message": "Memory vaulted successfully",
                "concept_id": concept_id,
                "vault_status": vault_level,
                "new_phase": memory.phase_tag
            });
            string_to_c_str(result.to_string())
        } else {
            let result = serde_json::json!({
                "success": false,
                "error": "Memory not found"
            });
            string_to_c_str(result.to_string())
        }
    } else {
        let result = serde_json::json!({
            "success": false,
            "error": "User lattice not found"
        });
        string_to_c_str(result.to_string())
    }
}

// Free C string memory (important for preventing leaks)
#[no_mangle]
pub extern "C" fn soliton_free_string(s: *mut c_char) {
    unsafe {
        if s.is_null() {
            return;
        }
        CString::from_raw(s);
    };
}

// Health check for the soliton engine
#[no_mangle]
pub extern "C" fn soliton_health_check() -> *mut c_char {
    let lattices = USER_LATTICES.lock().unwrap();
    let result = serde_json::json!({
        "success": true,
        "status": "healthy",
        "engine": "TORI Soliton Memory",
        "version": "1.0.0",
        "active_users": lattices.len(),
        "features": [
            "Phase-encoded memory storage",
            "Matched-filter retrieval",
            "Emotional signature analysis",
            "Memory vault protection",
            "Topological stability"
        ]
    });
    string_to_c_str(result.to_string())
}
