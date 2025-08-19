//! Safety modules for the concept mesh
//!
//! This directory contains safety-related modules for the concept mesh,
//! including stability analysis, certification, and constraints.

pub mod lyapunov_certifier;

// Re-export key components
pub use lyapunov_certifier::{
    CertificationResult, LyapunovCertifier, PlanEdge, PlanGraph, PlanNode, StabilityConfig,
};

/// Assess the stability of a plan cluster
///
/// This is a convenience wrapper around the LyapunovCertifier
/// for use in planner integration.
///
/// # Arguments
///
/// * `plan_graph` - The plan graph to certify
///
/// # Returns
///
/// * `CertificationResult` - The certification result
pub fn assess_cluster_stability(plan_graph: &PlanGraph) -> CertificationResult {
    use crate::lcn::LargeConceptNetwork;
    use std::sync::Arc;

    // Create LCN and certifier
    let lcn = Arc::new(LargeConceptNetwork::new());
    let certifier = LyapunovCertifier::new(lcn);

    // Perform certification
    certifier.certify_plan(plan_graph)
}

/// Create a stability certification error
///
/// This is a helper to create a standardized error when a plan
/// fails certification.
///
/// # Arguments
///
/// * `result` - The certification result
///
/// # Returns
///
/// * `String` - A formatted error message
pub fn create_certification_error(result: &CertificationResult) -> String {
    let mut error = String::from("Plan failed stability certification: ");

    if let Some(ref reason) = result.reason {
        error.push_str(reason);
    } else {
        error.push_str("unknown reason");
    }

    if let Some(ref recommendation) = result.recommendation {
        error.push_str("\nRecommendation: ");
        error.push_str(recommendation);
    }

    // Add metrics
    error.push_str("\nMetrics:");
    for (metric, value) in &result.metrics {
        error.push_str(&format!("\n  {}: {:.3}", metric, value));
    }

    error
}

/// UnsafePlanError type
#[derive(Debug)]
pub struct UnsafePlanError(pub String);

impl std::fmt::Display for UnsafePlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Unsafe plan: {}", self.0)
    }
}

impl std::error::Error for UnsafePlanError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assess_cluster_stability() {
        // Create a simple plan graph
        let plan = create_test_plan(3, 1, false);

        // Assess stability
        let result = assess_cluster_stability(&plan);

        // Should be stable
        assert!(result.cert_ok);
    }

    #[test]
    fn test_create_certification_error() {
        // Create a certification result
        let mut metrics = std::collections::HashMap::new();
        metrics.insert("instability".to_string(), 0.3);

        let result = CertificationResult {
            cert_ok: false,
            reason: Some("Instability risk".to_string()),
            metrics,
            recommendation: Some("Add constraints".to_string()),
        };

        // Create error message
        let error = create_certification_error(&result);

        // Check error content
        assert!(error.contains("Instability risk"));
        assert!(error.contains("Recommendation"));
        assert!(error.contains("instability: 0.300"));
    }

    // Helper to create test plans
    fn create_test_plan(node_count: usize, branching: usize, add_cycles: bool) -> PlanGraph {
        // Based on the lyapunov_certifier test implementation
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        let mut metadata = std::collections::HashMap::new();

        // Create nodes
        for i in 0..node_count {
            let mut attributes = std::collections::HashMap::new();
            attributes.insert("value".to_string(), format!("{}", i));

            nodes.push(PlanNode {
                id: format!("node_{}", i),
                node_type: "TestNode".to_string(),
                attributes,
            });
        }

        // Create edges
        for i in 0..node_count - 1 {
            for b in 0..branching {
                let target = (i + b + 1) % node_count;
                if target != i {
                    edges.push(PlanEdge {
                        source: format!("node_{}", i),
                        target: format!("node_{}", target),
                        edge_type: "TestEdge".to_string(),
                        weight: 1.0,
                    });
                }
            }
        }

        // Add cycles if requested
        if add_cycles {
            edges.push(PlanEdge {
                source: format!("node_{}", node_count - 1),
                target: "node_0".to_string(),
                edge_type: "CycleEdge".to_string(),
                weight: 1.0,
            });
        }

        // Add metadata
        metadata.insert("plan_type".to_string(), "test".to_string());
        metadata.insert("creator".to_string(), "test_suite".to_string());

        PlanGraph {
            nodes,
            edges,
            metadata,
        }
    }
}
