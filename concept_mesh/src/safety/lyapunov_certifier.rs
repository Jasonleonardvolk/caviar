//! Lyapunov Certifier for Plan Stability
//!
//! This module provides a stability certification system for plans based on
//! Lyapunov stability analysis. It ensures that plans won't lead to unstable
//! states by certifying their safety before execution.

use crate::diff::{ConceptDiff, ConceptDiffBuilder};
use crate::lcn::LargeConceptNetwork;

use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, error, info, warn};

/// Stability certification result
#[derive(Debug, Clone)]
pub struct CertificationResult {
    /// Whether the certification passed
    pub cert_ok: bool,

    /// Reason for failure (if cert_ok is false)
    pub reason: Option<String>,

    /// Stability metrics
    pub metrics: HashMap<String, f64>,

    /// Recommended action if certification failed
    pub recommendation: Option<String>,
}

/// Plan representation for certification
#[derive(Debug, Clone)]
pub struct PlanGraph {
    /// Nodes in the plan graph
    pub nodes: Vec<PlanNode>,

    /// Edges in the plan graph
    pub edges: Vec<PlanEdge>,

    /// Plan metadata
    pub metadata: HashMap<String, String>,
}

/// Node in the plan graph
#[derive(Debug, Clone)]
pub struct PlanNode {
    /// Node ID
    pub id: String,

    /// Node type
    pub node_type: String,

    /// Node attributes
    pub attributes: HashMap<String, String>,
}

/// Edge in the plan graph
#[derive(Debug, Clone)]
pub struct PlanEdge {
    /// Source node ID
    pub source: String,

    /// Target node ID
    pub target: String,

    /// Edge type
    pub edge_type: String,

    /// Edge weight
    pub weight: f64,
}

/// Stability threshold configuration
#[derive(Debug, Clone)]
pub struct StabilityConfig {
    /// Maximum allowed instability value
    pub max_instability: f64,

    /// Maximum allowed complexity
    pub max_complexity: f64,

    /// Minimum confidence for certification
    pub min_confidence: f64,

    /// Whether to use strict mode (fail on any warning)
    pub strict_mode: bool,
}

impl Default for StabilityConfig {
    fn default() -> Self {
        Self {
            max_instability: 0.2,
            max_complexity: 0.8,
            min_confidence: 0.7,
            strict_mode: false,
        }
    }
}

/// Lyapunov Certifier for plan stability
#[derive(Debug)]
pub struct LyapunovCertifier {
    /// Configuration for stability thresholds
    config: StabilityConfig,

    /// Large Concept Network for stability assessment
    lcn: Arc<LargeConceptNetwork>,
}

impl LyapunovCertifier {
    /// Create a new Lyapunov Certifier
    pub fn new(lcn: Arc<LargeConceptNetwork>) -> Self {
        Self {
            config: StabilityConfig::default(),
            lcn,
        }
    }

    /// Create a Lyapunov Certifier with custom configuration
    pub fn with_config(lcn: Arc<LargeConceptNetwork>, config: StabilityConfig) -> Self {
        Self { config, lcn }
    }

    /// Certify a plan for stability
    pub fn certify_plan(&self, plan: &PlanGraph) -> CertificationResult {
        debug!(
            "Certifying plan with {} nodes and {} edges",
            plan.nodes.len(),
            plan.edges.len()
        );

        // Analysis metrics
        let mut metrics = HashMap::new();

        // Step 1: Calculate plan complexity
        let complexity = self.calculate_plan_complexity(plan);
        metrics.insert("complexity".to_string(), complexity);

        // Step 2: Calculate stability derivative (V̇)
        let instability = self.calculate_stability_derivative(plan);
        metrics.insert("instability".to_string(), instability);

        // Step 3: Calculate confidence
        let confidence = self.calculate_confidence(plan);
        metrics.insert("confidence".to_string(), confidence);

        // Check stability thresholds
        let mut cert_ok = true;
        let mut reason = None;
        let mut recommendation = None;

        // Check instability (V̇)
        if instability > self.config.max_instability {
            cert_ok = false;
            reason = Some(format!(
                "Instability risk: V̇ = {:.3} > threshold {:.3}",
                instability, self.config.max_instability
            ));
            recommendation = Some(
                "Reduce the branching factor of the plan or add stabilizing constraints"
                    .to_string(),
            );
        }

        // Check complexity
        if complexity > self.config.max_complexity {
            cert_ok = false;
            reason = Some(format!(
                "Plan too complex: complexity = {:.3} > threshold {:.3}",
                complexity, self.config.max_complexity
            ));
            recommendation = Some(
                "Simplify the plan by reducing the number of steps or parallel paths".to_string(),
            );
        }

        // Check confidence
        if confidence < self.config.min_confidence {
            if self.config.strict_mode {
                cert_ok = false;
                reason = Some(format!(
                    "Low confidence: confidence = {:.3} < threshold {:.3}",
                    confidence, self.config.min_confidence
                ));
                recommendation =
                    Some("Add more constraints or explicit safety checks to the plan".to_string());
            } else {
                // In non-strict mode, low confidence is just a warning
                warn!(
                    "Low confidence in plan stability: {:.3} < threshold {:.3}",
                    confidence, self.config.min_confidence
                );
            }
        }

        if cert_ok {
            info!("Plan certified as stable with metrics: complexity={:.3}, instability={:.3}, confidence={:.3}", 
                 complexity, instability, confidence);
        } else {
            error!("Plan certification failed: {}", reason.as_ref().unwrap());
        }

        CertificationResult {
            cert_ok,
            reason,
            metrics,
            recommendation,
        }
    }

    /// Calculate the complexity of a plan
    fn calculate_plan_complexity(&self, plan: &PlanGraph) -> f64 {
        // Simple complexity metric: normalized by node and edge count
        // In a real implementation, this would use more sophisticated measures

        let node_count = plan.nodes.len() as f64;
        let edge_count = plan.edges.len() as f64;

        if node_count == 0.0 {
            return 0.0;
        }

        // Calculate edge-to-node ratio (higher means more complex)
        let edge_node_ratio = edge_count / node_count;

        // Calculate branching factor
        let max_branches = plan
            .nodes
            .iter()
            .map(|node| {
                plan.edges
                    .iter()
                    .filter(|edge| edge.source == node.id)
                    .count()
            })
            .max()
            .unwrap_or(0) as f64;

        // Normalize to 0-1 range
        let complexity = (0.5 * edge_node_ratio + 0.5 * (max_branches / 5.0)).min(1.0);

        debug!(
            "Plan complexity: {:.3} (nodes={}, edges={}, max_branches={})",
            complexity, node_count, edge_count, max_branches
        );

        complexity
    }

    /// Calculate the stability derivative (V̇) for a plan
    /// This is the heart of Lyapunov stability analysis
    fn calculate_stability_derivative(&self, plan: &PlanGraph) -> f64 {
        // In a real implementation, this would use actual Lyapunov analysis
        // For Day 1, we'll simulate with a heuristic approach

        // 1. Count potentially unstable nodes (high branching factor)
        let unstable_nodes = plan
            .nodes
            .iter()
            .filter(|node| {
                let out_edges = plan
                    .edges
                    .iter()
                    .filter(|edge| edge.source == node.id)
                    .count();

                // Nodes with many outgoing edges are potentially unstable
                out_edges > 3
            })
            .count() as f64;

        // 2. Count cycles in the graph (can lead to instability)
        let cycle_count = self.detect_cycles(plan) as f64;

        // 3. Check for isolated components (can indicate poor constraints)
        let isolated_components = self.count_components(plan) as f64;

        // Combine factors with weights
        let node_factor = unstable_nodes / plan.nodes.len().max(1) as f64;
        let cycle_factor = (cycle_count * 0.2).min(1.0);
        let isolation_factor = ((isolated_components - 1.0) * 0.3).max(0.0).min(1.0);

        let instability =
            (0.4 * node_factor + 0.4 * cycle_factor + 0.2 * isolation_factor).min(1.0);

        debug!(
            "Plan instability (V̇): {:.3} (unstable_nodes={}, cycles={}, components={})",
            instability, unstable_nodes, cycle_count, isolated_components
        );

        instability
    }

    /// Calculate confidence in the stability assessment
    fn calculate_confidence(&self, plan: &PlanGraph) -> f64 {
        // In a real implementation, this would assess the certainty of our stability prediction
        // For Day 1, we'll use a simple heuristic

        // More nodes/edges means less confidence (harder to predict)
        let size_factor = (1.0 - (plan.nodes.len() as f64 / 50.0).min(1.0));

        // Plans with more metadata provide more information, increasing confidence
        let metadata_factor = (plan.metadata.len() as f64 / 10.0).min(1.0);

        // Node detail level increases confidence
        let detail_factor = plan
            .nodes
            .iter()
            .map(|node| node.attributes.len())
            .sum::<usize>() as f64
            / (plan.nodes.len().max(1) * 5) as f64;

        let confidence = (0.4 * size_factor + 0.3 * metadata_factor + 0.3 * detail_factor).min(1.0);

        debug!("Plan stability confidence: {:.3}", confidence);

        confidence
    }

    /// Detect cycles in the plan graph
    /// Returns the number of cycles detected
    fn detect_cycles(&self, plan: &PlanGraph) -> usize {
        // Simple cycle detection using DFS
        // In a real implementation, we'd use a proper graph algorithm library

        // Build adjacency list
        let mut adj_list: HashMap<String, Vec<String>> = HashMap::new();

        for edge in &plan.edges {
            adj_list
                .entry(edge.source.clone())
                .or_insert_with(Vec::new)
                .push(edge.target.clone());
        }

        // Track visited nodes
        let mut visited = HashMap::new();
        let mut rec_stack = HashMap::new();
        let mut cycle_count = 0;

        // DFS from each node
        for node in &plan.nodes {
            if !visited.contains_key(&node.id) {
                if self.detect_cycles_util(&node.id, &adj_list, &mut visited, &mut rec_stack) {
                    cycle_count += 1;
                }
            }
        }

        cycle_count
    }

    /// Utility function for cycle detection
    fn detect_cycles_util(
        &self,
        node: &str,
        adj_list: &HashMap<String, Vec<String>>,
        visited: &mut HashMap<String, bool>,
        rec_stack: &mut HashMap<String, bool>,
    ) -> bool {
        // Mark node as visited and add to recursion stack
        visited.insert(node.to_string(), true);
        rec_stack.insert(node.to_string(), true);

        // Visit neighbors
        if let Some(neighbors) = adj_list.get(node) {
            for neighbor in neighbors {
                // If not visited, recurse
                if !visited.contains_key(neighbor) {
                    if self.detect_cycles_util(neighbor, adj_list, visited, rec_stack) {
                        return true;
                    }
                }
                // If in recursion stack, cycle detected
                else if rec_stack.get(neighbor).cloned().unwrap_or(false) {
                    return true;
                }
            }
        }

        // Remove from recursion stack
        rec_stack.insert(node.to_string(), false);
        false
    }

    /// Count connected components in the plan graph
    fn count_components(&self, plan: &PlanGraph) -> usize {
        // Simple component counting using DFS
        // In a real implementation, we'd use a proper graph algorithm library

        // Build adjacency list (bidirectional)
        let mut adj_list: HashMap<String, Vec<String>> = HashMap::new();

        for edge in &plan.edges {
            adj_list
                .entry(edge.source.clone())
                .or_insert_with(Vec::new)
                .push(edge.target.clone());

            adj_list
                .entry(edge.target.clone())
                .or_insert_with(Vec::new)
                .push(edge.source.clone());
        }

        // Track visited nodes
        let mut visited = HashMap::new();
        let mut component_count = 0;

        // DFS from each unvisited node
        for node in &plan.nodes {
            if !visited.contains_key(&node.id) {
                self.dfs(&node.id, &adj_list, &mut visited);
                component_count += 1;
            }
        }

        component_count
    }

    /// DFS utility for component counting
    fn dfs(
        &self,
        node: &str,
        adj_list: &HashMap<String, Vec<String>>,
        visited: &mut HashMap<String, bool>,
    ) {
        // Mark node as visited
        visited.insert(node.to_string(), true);

        // Visit neighbors
        if let Some(neighbors) = adj_list.get(node) {
            for neighbor in neighbors {
                if !visited.contains_key(neighbor) {
                    self.dfs(neighbor, adj_list, visited);
                }
            }
        }
    }

    /// Create a !PlanRejected ConceptDiff
    pub fn create_plan_rejected_diff(
        &self,
        plan_id: &str,
        result: &CertificationResult,
    ) -> ConceptDiff {
        let mut attributes = HashMap::new();
        attributes.insert("plan_id".to_string(), plan_id.to_string());

        if let Some(ref reason) = result.reason {
            attributes.insert("reason".to_string(), reason.clone());
        }

        for (metric, value) in &result.metrics {
            attributes.insert(format!("metric_{}", metric), format!("{:.6}", value));
        }

        if let Some(ref recommendation) = result.recommendation {
            attributes.insert("recommendation".to_string(), recommendation.clone());
        }

        // Create PlanRejected diff
        let builder = ConceptDiffBuilder::new(0) // Frame ID will be assigned later
            .create_with_attributes(
                format!("plan_rejection_{}", plan_id),
                "PlanRejection",
                attributes,
            );

        builder.build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stable_plan() {
        let lcn = Arc::new(LargeConceptNetwork::new());
        let certifier = LyapunovCertifier::new(Arc::clone(&lcn));

        // Create a simple stable plan
        let plan = create_test_plan(3, 2, false);

        let result = certifier.certify_plan(&plan);

        assert!(result.cert_ok);
        assert!(result.reason.is_none());
        assert!(result.metrics.contains_key("instability"));
        assert!(result.metrics.contains_key("complexity"));
    }

    #[test]
    fn test_unstable_plan() {
        let lcn = Arc::new(LargeConceptNetwork::new());
        let certifier = LyapunovCertifier::new(Arc::clone(&lcn));

        // Create a complex unstable plan
        let plan = create_test_plan(10, 5, true);

        let result = certifier.certify_plan(&plan);

        assert!(!result.cert_ok);
        assert!(result.reason.is_some());
        assert!(result.recommendation.is_some());
    }

    // Helper to create test plans
    fn create_test_plan(node_count: usize, branching: usize, add_cycles: bool) -> PlanGraph {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        let mut metadata = HashMap::new();

        // Create nodes
        for i in 0..node_count {
            let mut attributes = HashMap::new();
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
