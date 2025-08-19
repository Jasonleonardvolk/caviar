//! Phase Event Bus
//!
//! A high-performance pub/sub event system that synchronizes agents and components
//! with a 1kHz clock. The Phase Event Bus enables phase-coherent communication 
//! between mesh nodes and ensures stability through controlled message propagation.

use crate::mesh::MeshNode;

use async_trait::async_trait;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use tokio::sync::mpsc::{self, Receiver, Sender};
use tokio::time;
use tracing::{debug, error, info, trace, warn};

/// Phase event types
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PhaseEventType {
    /// Global tick event (sent every 1ms)
    Tick,
    
    /// Phase update from an agent
    PhaseUpdate,
    
    /// Phase drift detected
    PhaseDrift,
    
    /// Phase resynchronization
    PhaseResync,
    
    /// Concept-related events
    ConceptCreated,
    ConceptUpdated,
    ConceptLinked,
    ConceptRemoved,
    
    /// Agent status events
    AgentOnline,
    AgentOffline,
    AgentBusy,
    AgentIdle,
    
    /// Plan-related events
    PlanProposed,
    PlanApproved,
    PlanRejected,
    PlanExecuted,
    
    /// Custom event
    Custom(String),
}

impl fmt::Display for PhaseEventType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PhaseEventType::Tick => write!(f, "Tick"),
            PhaseEventType::PhaseUpdate => write!(f, "PhaseUpdate"),
            PhaseEventType::PhaseDrift => write!(f, "PhaseDrift"),
            PhaseEventType::PhaseResync => write!(f, "PhaseResync"),
            PhaseEventType::ConceptCreated => write!(f, "ConceptCreated"),
            PhaseEventType::ConceptUpdated => write!(f, "ConceptUpdated"),
            PhaseEventType::ConceptLinked => write!(f, "ConceptLinked"),
            PhaseEventType::ConceptRemoved => write!(f, "ConceptRemoved"),
            PhaseEventType::AgentOnline => write!(f, "AgentOnline"),
            PhaseEventType::AgentOffline => write!(f, "AgentOffline"),
            PhaseEventType::AgentBusy => write!(f, "AgentBusy"),
            PhaseEventType::AgentIdle => write!(f, "AgentIdle"),
            PhaseEventType::PlanProposed => write!(f, "PlanProposed"),
            PhaseEventType::PlanApproved => write!(f, "PlanApproved"),
            PhaseEventType::PlanRejected => write!(f, "PlanRejected"),
            PhaseEventType::PlanExecuted => write!(f, "PlanExecuted"),
            PhaseEventType::Custom(name) => write!(f, "Custom:{}", name),
        }
    }
}

/// Phase event subscription pattern
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SubscriptionPattern {
    /// Event type
    pub event_type: Option<PhaseEventType>,
    
    /// Source agent ID
    pub source: Option<String>,
    
    /// Target agent ID
    pub target: Option<String>,
}

impl SubscriptionPattern {
    /// Create a new subscription pattern
    pub fn new(
        event_type: Option<PhaseEventType>, 
        source: Option<String>, 
        target: Option<String>
    ) -> Self {
        Self {
            event_type,
            source,
            target,
        }
    }
    
    /// Subscribe to all events
    pub fn all() -> Self {
        Self {
            event_type: None,
            source: None,
            target: None,
        }
    }
    
    /// Subscribe to a specific event type
    pub fn event(event_type: PhaseEventType) -> Self {
        Self {
            event_type: Some(event_type),
            source: None,
            target: None,
        }
    }
    
    /// Subscribe to events from a specific source
    pub fn from(source: &str) -> Self {
        Self {
            event_type: None,
            source: Some(source.to_string()),
            target: None,
        }
    }
    
    /// Subscribe to events targeting a specific agent
    pub fn to(target: &str) -> Self {
        Self {
            event_type: None,
            source: None,
            target: Some(target.to_string()),
        }
    }
    
    /// Subscribe to events of a specific type from a specific source
    pub fn event_from(event_type: PhaseEventType, source: &str) -> Self {
        Self {
            event_type: Some(event_type),
            source: Some(source.to_string()),
            target: None,
        }
    }
    
    /// Subscribe to events of a specific type targeting a specific agent
    pub fn event_to(event_type: PhaseEventType, target: &str) -> Self {
        Self {
            event_type: Some(event_type),
            source: None,
            target: Some(target.to_string()),
        }
    }
    
    /// Check if an event matches this subscription pattern
    pub fn matches(&self, event: &PhaseEvent) -> bool {
        // Match event type if specified
        if let Some(ref event_type) = self.event_type {
            if event.event_type != *event_type {
                return false;
            }
        }
        
        // Match source if specified
        if let Some(ref source) = self.source {
            if event.source != *source {
                return false;
            }
        }
        
        // Match target if specified
        if let Some(ref target) = self.target {
            if event.target.as_deref() != Some(target) {
                return false;
            }
        }
        
        true
    }
}

/// Phase event payload data
#[derive(Debug, Clone)]
pub enum PhaseEventData {
    /// No data
    None,
    
    /// Integer value
    Integer(i64),
    
    /// Float value
    Float(f64),
    
    /// String value
    String(String),
    
    /// Boolean value
    Boolean(bool),
    
    /// Phase value (0.0 - 1.0)
    Phase(f64),
    
    /// Phase and amplitude
    PhaseAmplitude(f64, f64),
    
    /// Concept ID
    ConceptId(String),
    
    /// List of concept IDs
    ConceptIds(Vec<String>),
    
    /// Agent ID
    AgentId(String),
    
    /// Plan ID
    PlanId(String),
    
    /// JSON data
    Json(serde_json::Value),
    
    /// Custom data
    Custom(Arc<dyn std::any::Any + Send + Sync>),
}

/// Phase event
#[derive(Debug, Clone)]
pub struct PhaseEvent {
    /// Event ID
    pub id: u64,
    
    /// Timestamp (milliseconds since start)
    pub timestamp: u64,
    
    /// Event type
    pub event_type: PhaseEventType,
    
    /// Source agent ID
    pub source: String,
    
    /// Target agent ID (if any)
    pub target: Option<String>,
    
    /// Event data
    pub data: PhaseEventData,
}

impl PhaseEvent {
    /// Create a new phase event
    pub fn new(
        id: u64,
        timestamp: u64,
        event_type: PhaseEventType,
        source: String,
        target: Option<String>,
        data: PhaseEventData,
    ) -> Self {
        Self {
            id,
            timestamp,
            event_type,
            source,
            target,
            data,
        }
    }
    
    /// Create a new tick event
    pub fn tick(timestamp: u64, source: &str) -> Self {
        Self {
            id: timestamp, // Use timestamp as ID for tick events
            timestamp,
            event_type: PhaseEventType::Tick,
            source: source.to_string(),
            target: None,
            data: PhaseEventData::Integer(timestamp as i64),
        }
    }
    
    /// Create a new phase update event
    pub fn phase_update(timestamp: u64, source: &str, phase: f64) -> Self {
        Self {
            id: timestamp, // Use timestamp as ID for now
            timestamp,
            event_type: PhaseEventType::PhaseUpdate,
            source: source.to_string(),
            target: None,
            data: PhaseEventData::Phase(phase),
        }
    }
    
    /// Create a new concept created event
    pub fn concept_created(timestamp: u64, source: &str, concept_id: &str) -> Self {
        Self {
            id: timestamp, // Use timestamp as ID for now
            timestamp,
            event_type: PhaseEventType::ConceptCreated,
            source: source.to_string(),
            target: None,
            data: PhaseEventData::ConceptId(concept_id.to_string()),
        }
    }
    
    /// Create a new agent online event
    pub fn agent_online(timestamp: u64, source: &str) -> Self {
        Self {
            id: timestamp,
            timestamp,
            event_type: PhaseEventType::AgentOnline,
            source: source.to_string(),
            target: None,
            data: PhaseEventData::AgentId(source.to_string()),
        }
    }
}

/// Phase event handler function
pub type EventHandlerFn = Box<dyn Fn(&PhaseEvent) + Send + Sync>;

/// Phase event bus subscription
struct Subscription {
    /// Subscriber ID
    id: String,
    
    /// Subscription patterns
    patterns: Vec<SubscriptionPattern>,
    
    /// Event handler
    handler: EventHandlerFn,
}

/// Event statistics
#[derive(Debug, Default, Clone)]
pub struct EventStats {
    /// Total events processed
    pub total_events: u64,
    
    /// Events by type
    pub events_by_type: HashMap<PhaseEventType, u64>,
    
    /// Events by source
    pub events_by_source: HashMap<String, u64>,
    
    /// Dropped events (due to queue full)
    pub dropped_events: u64,
    
    /// Average processing time (microseconds)
    pub avg_processing_time_us: f64,
    
    /// Maximum processing time (microseconds)
    pub max_processing_time_us: u64,
    
    /// Tick jitter (maximum deviation from expected 1ms period, microseconds)
    pub tick_jitter_us: u64,
}

/// Phase event bus configuration
#[derive(Debug, Clone)]
pub struct PhaseEventBusConfig {
    /// Enable tick events
    pub enable_tick: bool,
    
    /// Tick interval in milliseconds
    pub tick_interval_ms: u64,
    
    /// Queue capacity
    pub queue_capacity: usize,
    
    /// Message timeout (milliseconds)
    pub message_timeout_ms: u64,
    
    /// Enable statistics
    pub enable_stats: bool,
}

impl Default for PhaseEventBusConfig {
    fn default() -> Self {
        Self {
            enable_tick: true,
            tick_interval_ms: 1, // 1kHz by default
            queue_capacity: 1000,
            message_timeout_ms: 100,
            enable_stats: true,
        }
    }
}

/// Phase event bus implementation
pub struct PhaseEventBus {
    /// Bus ID
    id: String,
    
    /// Configuration
    config: PhaseEventBusConfig,
    
    /// Subscribers
    subscribers: Arc<RwLock<Vec<Subscription>>>,
    
    /// Event sender
    event_sender: Sender<PhaseEvent>,
    
    /// Event receiver
    event_receiver: Option<Receiver<PhaseEvent>>,
    
    /// Event counter
    event_counter: Arc<Mutex<u64>>,
    
    /// Start time
    start_time: Instant,
    
    /// Running flag
    running: Arc<RwLock<bool>>,
    
    /// Statistics
    stats: Arc<RwLock<EventStats>>,
}

impl PhaseEventBus {
    /// Create a new phase event bus
    pub fn new(id: &str) -> Self {
        Self::with_config(id, PhaseEventBusConfig::default())
    }
    
    /// Create a new phase event bus with custom configuration
    pub fn with_config(id: &str, config: PhaseEventBusConfig) -> Self {
        let (tx, rx) = mpsc::channel(config.queue_capacity);
        
        Self {
            id: id.to_string(),
            config,
            subscribers: Arc::new(RwLock::new(Vec::new())),
            event_sender: tx,
            event_receiver: Some(rx),
            event_counter: Arc::new(Mutex::new(0)),
            start_time: Instant::now(),
            running: Arc::new(RwLock::new(false)),
            stats: Arc::new(RwLock::new(EventStats::default())),
        }
    }
    
    /// Start the event bus
    pub async fn start(&mut self) -> Result<(), String> {
        {
            let mut running = self.running.write().unwrap();
            if *running {
                return Err("Event bus already running".to_string());
            }
            *running = true;
        }
        
        // Take the receiver
        let receiver = self.event_receiver.take()
            .ok_or_else(|| "Event receiver already taken".to_string())?;
        
        // Start the event loop
        let subscribers = Arc::clone(&self.subscribers);
        let running = Arc::clone(&self.running);
        let stats = Arc::clone(&self.stats);
        let config = self.config.clone();
        let bus_id = self.id.clone();
        let event_sender = self.event_sender.clone();
        let event_counter = Arc::clone(&self.event_counter);
        let start_time = self.start_time;
        
        tokio::spawn(async move {
            Self::event_loop(
                receiver, 
                subscribers, 
                running, 
                stats, 
                config, 
                bus_id, 
                event_sender, 
                event_counter, 
                start_time
            ).await;
        });
        
        info!("Phase event bus {} started", self.id);
        Ok(())
    }
    
    /// Stop the event bus
    pub async fn stop(&self) -> Result<(), String> {
        let mut running = self.running.write().unwrap();
        if !*running {
            return Err("Event bus not running".to_string());
        }
        *running = false;
        
        info!("Phase event bus {} stopped", self.id);
        Ok(())
    }
    
    /// Get a reference to the event sender
    pub fn get_sender(&self) -> Sender<PhaseEvent> {
        self.event_sender.clone()
    }
    
    /// Subscribe to events
    pub fn subscribe<F>(&self, subscriber_id: &str, patterns: Vec<SubscriptionPattern>, handler: F) -> Result<(), String>
    where
        F: Fn(&PhaseEvent) + Send + Sync + 'static,
    {
        let mut subscribers = self.subscribers.write().unwrap();
        
        // Check if subscriber already exists
        if subscribers.iter().any(|s| s.id == subscriber_id) {
            return Err(format!("Subscriber {} already exists", subscriber_id));
        }
        
        // Add subscription
        subscribers.push(Subscription {
            id: subscriber_id.to_string(),
            patterns,
            handler: Box::new(handler),
        });
        
        debug!("Subscriber {} added to event bus {}", subscriber_id, self.id);
        Ok(())
    }
    
    /// Unsubscribe from events
    pub fn unsubscribe(&self, subscriber_id: &str) -> Result<(), String> {
        let mut subscribers = self.subscribers.write().unwrap();
        
        // Find subscriber
        let index = subscribers.iter().position(|s| s.id == subscriber_id)
            .ok_or_else(|| format!("Subscriber {} not found", subscriber_id))?;
        
        // Remove subscription
        subscribers.remove(index);
        
        debug!("Subscriber {} removed from event bus {}", subscriber_id, self.id);
        Ok(())
    }
    
    /// Publish an event to the bus
    pub async fn publish(&self, event: PhaseEvent) -> Result<(), String> {
        // Send the event
        if let Err(e) = self.event_sender.send(event).await {
            error!("Failed to send event: {}", e);
            
            // Update stats
            if self.config.enable_stats {
                let mut stats = self.stats.write().unwrap();
                stats.dropped_events += 1;
            }
            
            return Err(format!("Failed to send event: {}", e));
        }
        
        Ok(())
    }
    
    /// Publish a simple event
    pub async fn publish_simple(
        &self,
        event_type: PhaseEventType,
        source: &str,
        target: Option<&str>,
        data: PhaseEventData,
    ) -> Result<(), String> {
        // Get the next event ID
        let id = {
            let mut counter = self.event_counter.lock().unwrap();
            *counter += 1;
            *counter
        };
        
        // Create the event
        let timestamp = self.get_timestamp();
        let event = PhaseEvent::new(
            id,
            timestamp,
            event_type,
            source.to_string(),
            target.map(|t| t.to_string()),
            data,
        );
        
        // Publish the event
        self.publish(event).await
    }
    
    /// Get current timestamp (milliseconds since start)
    pub fn get_timestamp(&self) -> u64 {
        let elapsed = self.start_time.elapsed();
        elapsed.as_secs() * 1000 + u64::from(elapsed.subsec_millis())
    }
    
    /// Get statistics
    pub fn get_stats(&self) -> EventStats {
        let stats = self.stats.read().unwrap();
        stats.clone()
    }
    
    /// Reset statistics
    pub fn reset_stats(&self) {
        let mut stats = self.stats.write().unwrap();
        *stats = EventStats::default();
    }
    
    /// Check if the event bus is running
    pub fn is_running(&self) -> bool {
        let running = self.running.read().unwrap();
        *running
    }
    
    /// Get the number of subscribers
    pub fn subscriber_count(&self) -> usize {
        let subscribers = self.subscribers.read().unwrap();
        subscribers.len()
    }
    
    /// Main event loop
    async fn event_loop(
        mut receiver: Receiver<PhaseEvent>,
        subscribers: Arc<RwLock<Vec<Subscription>>>,
        running: Arc<RwLock<bool>>,
        stats: Arc<RwLock<EventStats>>,
        config: PhaseEventBusConfig,
        bus_id: String,
        event_sender: Sender<PhaseEvent>,
        event_counter: Arc<Mutex<u64>>,
        start_time: Instant,
    ) {
        // Start the tick task if enabled
        let mut tick_task = if config.enable_tick {
            let tick_interval = Duration::from_millis(config.tick_interval_ms);
            let event_sender = event_sender.clone();
            let event_counter = Arc::clone(&event_counter);
            let running = Arc::clone(&running);
            let bus_id = bus_id.clone();
            let start_time = start_time;
            
            let task = tokio::spawn(async move {
                let mut interval = time::interval(tick_interval);
                
                loop {
                    interval.tick().await;
                    
                    // Check if still running
                    if !*running.read().unwrap() {
                        break;
                    }
                    
                    // Get the next event ID
                    let id = {
                        let mut counter = event_counter.lock().unwrap();
                        *counter += 1;
                        *counter
                    };
                    
                    // Calculate timestamp
                    let elapsed = start_time.elapsed();
                    let timestamp = elapsed.as_secs() * 1000 + u64::from(elapsed.subsec_millis());
                    
                    // Create a tick event
                    let event = PhaseEvent::new(
                        id,
                        timestamp,
                        PhaseEventType::Tick,
                        bus_id.clone(),
                        None,
                        PhaseEventData::Integer(timestamp as i64),
                    );
                    
                    // Send the event
                    if let Err(e) = event_sender.send(event).await {
                        error!("Failed to send tick event: {}", e);
                        break;
                    }
                }
            });
            
            Some(task)
        } else {
            None
        };
        
        // Process events
        let mut last_tick_time = Instant::now();
        
        while let Some(event) = receiver.recv().await {
            // Check if still running
            if !*running.read().unwrap() {
                break;
            }
            
            // Record processing start time
            let process_start = Instant::now();
            
            // Measure tick jitter if this is a tick event
            if event.event_type == PhaseEventType::Tick && config.enable_stats {
                let elapsed = process_start.duration_since(last_tick_time);
                let jitter = elapsed.as_micros().saturating_sub(config.tick_interval_ms as u128 * 1000);
                
                let mut stats = stats.write().unwrap();
                stats.tick_jitter_us = stats.tick_jitter_us.max(jitter as u64);
                
                last_tick_time = process_start;
            }
            
            // Get relevant subscribers
            let mut relevant_subscribers = Vec::new();
            {
                let subscribers = subscribers.read().unwrap();
                
                for subscriber in subscribers.iter() {
                    for pattern in &subscriber.patterns {
                        if pattern.matches(&event) {
                            relevant_subscribers.push(subscriber);
                            break;
                        }
                    }
                }
            }
            
            // Deliver the event to subscribers
            for subscriber in relevant_subscribers {
                (subscriber.handler)(&event);
            }
            
            // Update statistics
            if config.enable_stats {
                let process_time = process_start.elapsed().as_micros() as u64;
                
                let mut stats = stats.write().unwrap();
                stats.total_events += 1;
                
                // Update event type count
                *stats.events_by_type.entry(event.event_type.clone()).or_insert(0) += 1;
                
                // Update source count
                *stats.events_by_source.entry(event.source.clone()).or_insert(0) += 1;
                
                // Update processing time stats
                stats.max_processing_time_us = stats.max_processing_time_us.max(process_time);
                
                // Update average processing time
                if stats.total_events == 1 {
                    stats.avg_processing_time_us = process_time as f64;
                } else {
                    let alpha = 0.01; // Exponential moving average factor
                    stats.avg_processing_time_us = 
                        (1.0 - alpha) * stats.avg_processing_time_us + 
                        alpha * process_time as f64;
                }
            }
            
            // If processing took too long, log a warning
            let process_time = process_start.elapsed();
            if process_time > Duration::from_millis(config.tick_interval_ms) {
                warn!(
                    "Event processing took {} ms, which is longer than the tick interval ({} ms)",
                    process_time.as_millis(),
                    config.tick_interval_ms
                );
            }
        }
        
        // Cancel tick task if it exists
        if let Some(task) = tick_task.take() {
            task.abort();
        }
        
        info!("Event loop for bus {} stopped", bus_id);
    }
}

/// Phase agent trait
#[async_trait]
pub trait PhaseAgent: Send + Sync {
    /// Get the agent ID
    fn id(&self) -> &str;
    
    /// Initialize the agent
    async fn initialize(&mut self, bus: &PhaseEventBus) -> Result<(), String>;
    
    /// Handle a phase event
    fn on_event(&mut self, event: &PhaseEvent);
    
    /// Emit a phase update
    async fn emit_phase(&self, bus: &PhaseEventBus, phase: f64) -> Result<(), String> {
        let timestamp = bus.get_timestamp();
        let event = PhaseEvent::phase_update(timestamp, self.id(), phase);
        bus.publish(event).await
    }
    
    /// Shutdown the agent
    async fn shutdown(&mut self) -> Result<(), String>;
}

/// Create a new phase event bus
pub async fn create_phase_event_bus(id: &str) -> Result<Arc<PhaseEventBus>, String> {
    let mut bus = PhaseEventBus::new(id);
    bus.start().await?;
    Ok(Arc::new(bus))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;
    
    // Test helpers
    struct TestAgent {
        id: String,
        received_events: Arc<Mutex<Vec<PhaseEvent>>>,
        initialized: AtomicBool,
    }
    
    impl TestAgent {
        fn new(id: &str) -> Self {
            Self {
                id: id.to_string(),
                received_events: Arc::new(Mutex::new(Vec::new())),
                initialized: AtomicBool::new(false),
            }
        }
        
        fn get_received_events(&self) -> Vec<PhaseEvent> {
            let events = self.received_events.lock().unwrap();
            events.clone()
        }
    }
    
    #[async_trait]
    impl PhaseAgent for TestAgent {
        fn id(&self) -> &str {
            &self.id
        }
        
        async fn initialize(&mut self, _bus: &PhaseEventBus) -> Result<(), String> {
            self.initialized.store(true, Ordering::SeqCst);
            Ok(())
        }
        
        fn on_event(&mut self, event: &PhaseEvent) {
            let mut events = self.received_events.lock().unwrap();
            events.push(event.clone());
        }
        
        async fn shutdown(&mut self) -> Result<(), String> {
            Ok(())
        }
    }
    
    #[tokio::test]
    async fn test_subscription_pattern_matching() {
        // Create a test event
        let event = PhaseEvent::new(
            1,
            100,
            PhaseEventType::ConceptCreated,
            "agent1".to_string(),
            Some("agent2".to_string()),
            PhaseEventData::String("test".to_string()),
        );
        
        // Test all pattern
        let pattern = SubscriptionPattern::all();
        assert!(pattern.matches(&event));
        
        // Test event pattern
        let pattern = SubscriptionPattern::event(PhaseEventType::ConceptCreated);
        assert!(pattern.matches(&event));
        
        let pattern = SubscriptionPattern::event(PhaseEventType::ConceptUpdated);
        assert!(!pattern.matches(&event));
        
        // Test source pattern
        let pattern = SubscriptionPattern::from("agent1");
        assert!(pattern.matches(&event));
        
        let pattern = SubscriptionPattern::from("agent3");
        assert!(!pattern.matches(&event));
        
        // Test target pattern
        let pattern = SubscriptionPattern::to("agent2");
        assert!(pattern.matches(&event));
        
        let pattern = SubscriptionPattern::to("agent3");
        assert!(!pattern.matches(&event));
        
        // Test combined patterns
        let pattern = SubscriptionPattern::event_from(PhaseEventType::ConceptCreated, "agent1");
        assert!(pattern.matches(&event));
        
        let pattern = SubscriptionPattern::event_from(PhaseEventType::ConceptUpdated, "agent1");
        assert!(!pattern.matches(&event));
        
        let pattern = SubscriptionPattern::event_to(PhaseEventType::ConceptCreated, "agent2");
        assert!(pattern.matches(&event));
        
        let pattern = SubscriptionPattern::event_to(PhaseEventType::ConceptCreated, "agent3");
        assert!(!pattern.matches(&event));
    }
    
    #[tokio::test]
    async fn test_publish_and_subscribe() {
        // Create event bus
        let mut bus = PhaseEventBus::with_config(
            "test", 
            PhaseEventBusConfig {
                enable_tick: false, // Disable tick events for this test
                ..PhaseEventBusConfig::default()
            }
        );
        
        bus.start().await.unwrap();
        
        // Create a test agent
        let agent = Arc::new(Mutex::new(TestAgent::new("agent1")));
        
        // Subscribe to events
        let agent_clone = Arc::clone(&agent);
        bus.subscribe(
            "agent1",
            vec![SubscriptionPattern::all()],
            move |event| {
                let mut agent = agent_clone.lock().unwrap();
                agent.on_event(event);
            },
        ).unwrap();
        
        // Publish a test event
        let event = PhaseEvent::new(
            1,
            100,
            PhaseEventType::ConceptCreated,
            "test".to_string(),
            None,
            PhaseEventData::String("test".to_string()),
        );
        
        bus.publish(event.clone()).await.unwrap();
        
        // Wait for the event to be processed
        tokio::time::sleep(Duration::from_millis(10)).await;
        
        // Check that the agent received the event
        let agent = agent.lock().unwrap();
        let events = agent.get_received_events();
        
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].id, event.id);
        assert_eq!(events[0].event_type, event.event_type);
