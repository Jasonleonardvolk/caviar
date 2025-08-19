//! Planner module for the concept mesh
//!
//! This module implements a planning agent that creates and validates execution plans
//! for tasks in the concept mesh. Plans are certified for stability using Lyapunov
//! analysis before execution.

use crate::diff::{ConceptDiff, ConceptDiffBuilder};
use crate::lcn::LargeConceptNetwork;
use crate::safety::{
    assess_cluster_stability, CertificationResult, PlanGraph, PlanNode, PlanEdge,
    StabilityConfig, UnsafePlanError
};

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tracing::{info, error, debug, warn};

/// Plan status enum
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanStatus {
    /// Plan is being created
    Creating,
    
    /// Plan is being certified for stability
    Certifying,
    
    /// Plan was rejected due to instability
    Rejected,
    
    /// Plan is being executed
    Executing,
    
    /// Plan execution completed successfully
    Completed,
    
    /// Plan execution failed
    Failed,
}

/// Plan structure representing a task execution plan
#[derive(Debug, Clone)]
pub struct Plan {
    /// Unique plan ID
    pub id: String,
    
    /// Task ID this plan is for
    pub task_id: String,
    
    /// Plan status
    pub status: PlanStatus,
    
    /// The actual plan graph
    pub graph: PlanGraph,
    
    /// Execution metadata
    pub metadata: HashMap<String, String>,
}

/// Planner configuration
#[derive(Debug, Clone)]
pub struct PlannerConfig {
    /// Stability thresholds
    pub stability_config: StabilityConfig,
    
    /// Maximum plan complexity
    pub max_plan_complexity: usize,
    
    /// Enable plan certification
    pub enable_certification: bool,
}

impl Default for PlannerConfig {
    fn default() -> Self {
        Self {
            stability_config: StabilityConfig::default(),
            max_plan_complexity: 100,
            enable_certification: true,
        }
    }
}

/// Planner agent for creating and executing plans
pub struct Planner {
    /// Planner configuration
    config: PlannerConfig,
    
    /// Planner LCN
    lcn: Arc<LargeConceptNetwork>,
    
    /// Active plans
    active_plans: Arc<Mutex<HashMap<String, Plan>>>,
}

impl Planner {
    /// Create a new planner
    pub fn new(lcn: Arc<LargeConceptNetwork>) -> Self {
        Self {
            config: PlannerConfig::default(),
            lcn,
            active_plans: Arc::new(Mutex::new(HashMap::new())),
        }
    }
    
    /// Create a planner with custom configuration
    pub fn with_config(lcn: Arc<LargeConceptNetwork>, config: PlannerConfig) -> Self {
        Self {
            config,
            lcn,
            active_plans: Arc::new(Mutex::new(HashMap::new())),
        }
    }
    
    /// Create a plan for a task
    pub fn create_plan(&self, task_id: &str, complexity: usize) -> Result<Plan, String> {
        debug!("Creating plan for task {}", task_id);
        
        // Check complexity limits
        if complexity > self.config.max_plan_complexity {
            return Err(format!(
                "Requested plan complexity {} exceeds maximum {}",
                complexity,
                self.config.max_plan_complexity
            ));
        }
        
        // Create a plan graph based on task requirements
        // In a real implementation, this would use more sophisticated planning
        let plan_graph = self.generate_plan_graph(task_id, complexity)?;
        
        // Create plan
        let plan_id = format!("plan_{}_{}",
                             task_id, 
                             chrono::Utc::now().timestamp());
        
        let plan = Plan {
            id: plan_id,
            task_id: task_id.to_string(),
            status: PlanStatus::Creating,
            graph: plan_graph,
            metadata: HashMap::new(),
        };
        
        // Store plan
        self.active_plans.lock().unwrap().insert(plan.id.clone(), plan.clone());
        
        Ok(plan)
    }
    
    /// Generate a plan graph for a task
    fn generate_plan_graph(&self, task_id: &str, complexity: usize) -> Result<PlanGraph, String> {
        // In a real implementation, this would use a real planner
        // For Day 1, we'll create a simple linear plan
        
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        let mut metadata = HashMap::new();
        
        // Create a linear plan with N steps
        let step_count = complexity.max(2);
        
        // Create nodes
        for i in 0..step_count {
            let node_id = format!("step_{}", i);
            let node_type = if i == 0 {
                "start"
            } else if i == step_count - 1 {
                "end"
            } else {
                "action"
            };
            
            let mut attributes = HashMap::new();
            attributes.insert("task_id".to_string(), task_id.to_string());
            attributes.insert("step".to_string(), i.to_string());
            
            nodes.push(PlanNode {
                id: node_id,
                node_type: node_type.to_string(),
                attributes,
            });
        }
        
        // Create edges (linear path)
        for i in 0..step_count - 1 {
            edges.push(PlanEdge {
                source: format!("step_{}", i),
                target: format!("step_{}", i + 1),
                edge_type: "sequence".to_string(),
                weight: 1.0,
            });
        }
        
        // Add metadata
        metadata.insert("task_id".to_string(), task_id.to_string());
        metadata.insert("complexity".to_string(), complexity.to_string());
        metadata.insert("generation_time".to_string(), chrono::Utc::now().to_rfc3339());
        
        Ok(PlanGraph {
            nodes,
            edges,
            metadata,
        })
    }
    
    /// Validate a plan for stability
    pub fn validate_plan(&self, plan_id: &str) -> Result<(), UnsafePlanError> {
        debug!("Validating plan {}", plan_id);
        
        // Get plan
        let mut plan = {
            let plans = self.active_plans.lock().unwrap();
            match plans.get(plan_id) {
                Some(p) => p.clone(),
                None => return Err(UnsafePlanError(format!("Plan not found: {}", plan_id))),
            }
        };
        
        // Skip certification if disabled
        if !self.config.enable_certification {
            info!("Plan certification disabled, skipping validation");
            self.update_plan_status(plan_id, PlanStatus::Certifying);
            return Ok(());
        }
        
        // Update status
        self.update_plan_status(plan_id, PlanStatus::Certifying);
        
        // Certify plan
        let cert_result = assess_cluster_stability(&plan.graph);
        
        // Check result
        if !cert_result.cert_ok {
            // Plan is rejected due to instability
            error!("Plan {} rejected due to instability", plan_id);
            
            // Update status
            self.update_plan_status(plan_id, PlanStatus::Rejected);
            
            // Create PlanRejected diff
            self.create_plan_rejected_diff(plan_id, &cert_result);
            
            // Return error
            return Err(UnsafePlanError(
                cert_result.reason.unwrap_or_else(|| "Unknown instability".to_string())
            ));
        }
        
        // Plan is certified
        info!("Plan {} certified as stable", plan_id);
        Ok(())
    }
    
    /// Create a plan rejection diff
    fn create_plan_rejected_diff(&self, plan_id: &str, result: &CertificationResult) {
        // Use the Lyapunov certifier to create a rejection diff
        let lcn = Arc::new(LargeConceptNetwork::new());
        let certifier = crate::safety::LyapunovCertifier::new(Arc::clone(&lcn));
        
        let diff = certifier.create_plan_rejected_diff(plan_id, result);
        
        // In a real implementation, we would:
        // 1. Assign a frame ID
        // 2. Record the diff to ψarc
        // 3. Apply it to the LCN
        
        // For Day 1, just log it
        info!("Created !PlanRejected diff for plan {}", plan_id);
        debug!("Diff: {:?}", diff);
    }
    
    /// Update plan status
    fn update_plan_status(&self, plan_id: &str, status: PlanStatus) {
        let mut plans = self.active_plans.lock().unwrap();
        if let Some(plan) = plans.get_mut(plan_id) {
            plan.status = status;
        }
    }
    
    /// Execute a plan
    pub async fn execute_plan(&self, plan_id: &str) -> Result<(), String> {
        debug!("Executing plan {}", plan_id);
        
        // Get plan
        let plan = {
            let plans = self.active_plans.lock().unwrap();
            match plans.get(plan_id) {
                Some(p) => p.clone(),
                None => return Err(format!("Plan not found: {}", plan_id)),
            }
        };
        
        // Check if plan is certified
        if plan.status == PlanStatus::Rejected {
            return Err(format!("Cannot execute rejected plan: {}", plan_id));
        }
        
        // Update status
        self.update_plan_status(plan_id, PlanStatus::Executing);
        
        // Execute plan (simulated)
        // In a real implementation, this would actually execute the plan
        
        // Simulate execution by sleeping
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        // Mark as completed
        self.update_plan_status(plan_id, PlanStatus::Completed);
        
        info!("Plan {} execution completed", plan_id);
        Ok(())
    }
    
    /// Get plan status
    pub fn get_plan_status(&self, plan_id: &str) -> Option<PlanStatus> {
        let plans = self.active_plans.lock().unwrap();
        plans.get(plan_id).map(|p| p.status.clone())
    }
}

/// Convenience function for validating plans from other modules
pub fn validate_plan(plan_graph: &PlanGraph) -> Result<(), UnsafePlanError> {
    // Perform certification
    let cert_result = assess_cluster_stability(plan_graph);
    
    // Check result
    if !cert_result.cert_ok {
        // Create error
        return Err(UnsafePlanError(
            cert_result.reason.unwrap_or_else(|| "Unknown instability".to_string())
        ));
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_create_plan() {
        let lcn = Arc::new(LargeConceptNetwork::new());
        let planner = Planner::new(Arc::clone(&lcn));
        
        let plan = planner.create_plan("test_task", 5).unwrap();
        
        assert_eq!(plan.status, PlanStatus::Creating);
        assert_eq!(plan.graph.nodes.len(), 5);
        assert_eq!(plan.graph.edges.len(), 4); // Linear plan, N-1 edges
    }
    
    #[test]
    fn test_validate_stable_plan() {
        let lcn = Arc::new(LargeConceptNetwork::new());
        let planner = Planner::new(Arc::clone(&lcn));
        
        let plan = planner.create_plan("test_task", 3).unwrap();
        
        let result = planner.validate_plan(&plan.id);
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_execute_plan() {
        let lcn = Arc::new(LargeConceptNetwork::new());
        let planner = Planner::new(Arc::clone(&lcn));
        
        let plan = planner.create_plan("test_task", 3).unwrap();
        planner.validate_plan(&plan.id).unwrap();
        
        let result = planner.execute_plan(&plan.id).await;
        assert!(result.is_ok());
        
        let status = planner.get_plan_status(&plan.id).unwrap();
        assert_eq!(status, PlanStatus::Completed);
    }
}
