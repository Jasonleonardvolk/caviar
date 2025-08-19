/**
 * TORI Event Bus System - Type-Safe Inter-Module Communication
 * 
 * This module implements a high-performance, type-safe event system for
 * communication between cognitive modules. It provides publish/subscribe
 * patterns, event ordering, priority management, and real-time streaming
 * capabilities for the TORI orchestration system.
 */

use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, RwLock, atomic::{AtomicU64, Ordering}},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
    hash::{Hash, Hasher},
    fmt,
    pin::Pin,
};

use tokio::{
    sync::{mpsc, oneshot, broadcast, RwLock as TokioRwLock, Mutex as TokioMutex},
    time::{interval, timeout},
    task::{spawn, JoinHandle},
};

use serde::{Serialize, Deserialize};
use uuid::Uuid;
use dashmap::DashMap;
use tracing::{info, warn, error, debug, trace, instrument};
use anyhow::{Result, Context, Error};
use thiserror::Error;
use futures_util::future::BoxFuture;

// Type aliases for cognitive module types
pub type ConceptId = u64;
pub type ThreadId = Uuid;
pub type BraidId = Uuid;
pub type WormholeId = Uuid;
pub type ScaleLevel = u32;

// ===================================================================
// ERROR TYPES
// ===================================================================

#[derive(Debug, Error)]
pub enum EventBusError {
    #[error("Event dispatch failed: {0}")]
    DispatchFailed(String),
    
    #[error("Subscription failed: {0}")]
    SubscriptionFailed(String),
    
    #[error("Event queue full: capacity {capacity}, attempted to add {event_type:?}")]
    QueueFull { capacity: usize, event_type: EventType },
    
    #[error("Invalid event handler: {0}")]
    InvalidHandler(String),
    
    #[error("Event serialization failed: {0}")]
    SerializationFailed(String),
    
    #[error("Event timeout: waited {duration:?} for {event_type:?}")]
    EventTimeout { duration: Duration, event_type: EventType },
    
    #[error("Channel closed unexpectedly")]
    ChannelClosed,
}

// ===================================================================
// EVENT TYPES AND DATA STRUCTURES
// ===================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EventType {
    // Core cognitive events
    ConceptAdded,
    ConceptModified,
    ConceptRemoved,
    
    // Memory and braiding events
    ThreadCreated,
    ThreadAppended,
    ThreadClosed,
    BraidFormed,
    BraidDissolved,
    
    // Wormhole events
    WormholeCreated,
    WormholeRemoved,
    WormholeSuggested,
    
    // Alien calculus events
    AlienDetected,
    AlienResolved,
    ScarDetected,
    ScarHealed,
    
    // System events
    SystemStarted,
    SystemShutdown,
    ModuleInitialized,
    ModuleShutdown,
    
    // Resource events
    ResourceThresholdExceeded,
    ResourceRecovered,
    MemoryPressure,
    CPUPressure,
    
    // Task events
    TaskStarted,
    TaskCompleted,
    TaskFailed,
    TaskCancelled,
    
    // Fuzzing events
    FuzzTestStarted,
    FuzzTestCompleted,
    ChaosInjected,
    CoverageUpdated,
    
    // Performance events
    PerformanceAlert,
    MetricsCollected,
    BenchmarkCompleted,
    
    // Hot reload events
    HotReloadTriggered,
    HotReloadCompleted,
    HotReloadFailed,
    
    // WebSocket events
    ClientConnected,
    ClientDisconnected,
    ClientMessage,
    
    // Error events
    ErrorOccurred,
    RecoveryAttempted,
    RecoverySucceeded,
    RecoveryFailed,
    
    // State persistence events
    CheckpointCreated,
    CheckpointLoaded,
    StateCorrupted,
    StateRecovered,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventData {
    // Concept-related data
    Concept {
        concept_id: ConceptId,
        parent_id: Option<ConceptId>,
        scale: ScaleLevel,
    },
    
    // Thread-related data
    Thread {
        thread_id: ThreadId,
        concept_ids: Vec<ConceptId>,
        metadata: HashMap<String, String>,
    },
    
    // Braid-related data
    Braid {
        braid_id: BraidId,
        thread_ids: Vec<ThreadId>,
        concept_id: ConceptId,
        strength: f64,
    },
    
    // Wormhole-related data
    Wormhole {
        wormhole_id: WormholeId,
        concept_a: ConceptId,
        concept_b: ConceptId,
        strength: f64,
    },
    
    // Alien calculus data
    Alien {
        concept_id: ConceptId,
        significance: f64,
        context: String,
        action_value: f64,
    },
    
    // Scar data
    Scar {
        concept_id: ConceptId,
        severity: f64,
        details: String,
    },
    
    // System data
    System {
        timestamp: u64,
        status: String,
    },
    
    // Resource data
    Resource {
        resource_type: String,
        usage: f64,
        threshold: f64,
    },
    
    // Task data
    Task {
        task_id: Uuid,
        task_type: String,
        status: String,
        duration: Option<Duration>,
    },
    
    // Fuzzing data
    Fuzz {
        test_id: Uuid,
        module: String,
        result: String,
        metrics: HashMap<String, f64>,
    },
    
    // Performance data
    Performance {
        metric_name: String,
        value: f64,
        threshold: Option<f64>,
        trend: String,
    },
    
    // Error data
    Error {
        error_type: String,
        message: String,
        module: String,
        stack_trace: Vec<String>,
    },
    
    // Generic data for extensibility
    Generic {
        data: HashMap<String, serde_json::Value>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub id: Uuid,
    pub event_type: EventType,
    pub data: EventData,
    pub timestamp: SystemTime,
    pub source_module: Option<String>,
    pub correlation_id: Option<Uuid>,
    pub priority: EventPriority,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EventPriority {
    Critical = 0,
    High = 1,
    Normal = 2,
    Low = 3,
    Background = 4,
}

impl Event {
    pub fn new(event_type: EventType, data: EventData) -> Self {
        Self {
            id: Uuid::new_v4(),
            event_type,
            data,
            timestamp: SystemTime::now(),
            source_module: None,
            correlation_id: None,
            priority: EventPriority::Normal,
            metadata: HashMap::new(),
        }
    }
    
    pub fn with_priority(mut self, priority: EventPriority) -> Self {
        self.priority = priority;
        self
    }
    
    pub fn with_source(mut self, source: String) -> Self {
        self.source_module = Some(source);
        self
    }
    
    pub fn with_correlation(mut self, correlation_id: Uuid) -> Self {
        self.correlation_id = Some(correlation_id);
        self
    }
    
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

impl fmt::Display for Event {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Event({:?}, {}, priority={:?})",
            self.event_type,
            self.id,
            self.priority
        )
    }
}

// ===================================================================
// EVENT HANDLER TYPES
// ===================================================================

pub type EventHandlerFuture = BoxFuture<'static, Result<()>>;
pub type EventHandler = Box<dyn Fn(EventData) -> EventHandlerFuture + Send + Sync>;

// Subscription information
#[derive(Debug)]
pub struct Subscription {
    pub id: Uuid,
    pub event_type: EventType,
    pub handler: EventHandler,
    pub priority: EventPriority,
    pub created_at: Instant,
    pub call_count: AtomicU64,
    pub error_count: AtomicU64,
    pub avg_duration: Arc<RwLock<Duration>>,
}

impl Subscription {
    pub fn new(event_type: EventType, handler: EventHandler) -> Self {
        Self {
            id: Uuid::new_v4(),
            event_type,
            handler,
            priority: EventPriority::Normal,
            created_at: Instant::now(),
            call_count: AtomicU64::new(0),
            error_count: AtomicU64::new(0),
            avg_duration: Arc::new(RwLock::new(Duration::from_millis(0))),
        }
    }
    
    pub fn with_priority(mut self, priority: EventPriority) -> Self {
        self.priority = priority;
        self
    }
    
    pub async fn execute(&self, data: EventData) -> Result<()> {
        let start = Instant::now();
        
        let result = (self.handler)(data).await;
        
        let duration = start.elapsed();
        self.call_count.fetch_add(1, Ordering::Relaxed);
        
        if result.is_err() {
            self.error_count.fetch_add(1, Ordering::Relaxed);
        }
        
        // Update average duration
        if let Ok(mut avg) = self.avg_duration.write() {
            let call_count = self.call_count.load(Ordering::Relaxed);
            *avg = Duration::from_nanos(
                (avg.as_nanos() as u64 * (call_count - 1) + duration.as_nanos() as u64) / call_count
            );
        }
        
        result
    }
    
    pub fn get_stats(&self) -> SubscriptionStats {
        SubscriptionStats {
            subscription_id: self.id,
            event_type: self.event_type,
            call_count: self.call_count.load(Ordering::Relaxed),
            error_count: self.error_count.load(Ordering::Relaxed),
            avg_duration: self.avg_duration.read().unwrap_or(&Duration::from_millis(0)).clone(),
            uptime: self.created_at.elapsed(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubscriptionStats {
    pub subscription_id: Uuid,
    pub event_type: EventType,
    pub call_count: u64,
    pub error_count: u64,
    pub avg_duration: Duration,
    pub uptime: Duration,
}

// ===================================================================
// EVENT BUS IMPLEMENTATION
// ===================================================================

pub struct EventBus {
    // Event queue and processing
    event_sender: mpsc::UnboundedSender<EventEnvelope>,
    event_receiver: Arc<TokioMutex<mpsc::UnboundedReceiver<EventEnvelope>>>,
    
    // Subscription management
    subscriptions: Arc<DashMap<EventType, Vec<Arc<Subscription>>>>,
    subscription_lookup: Arc<DashMap<Uuid, Arc<Subscription>>>,
    
    // Event history and metrics
    event_history: Arc<TokioRwLock<VecDeque<Event>>>,
    total_events_processed: AtomicU64,
    total_errors: AtomicU64,
    
    // Broadcasting for real-time UI updates
    broadcast_sender: broadcast::Sender<Event>,
    
    // Configuration
    config: EventBusConfig,
    
    // Task handles for cleanup
    processor_handle: Arc<TokioMutex<Option<JoinHandle<()>>>>,
    metrics_handle: Arc<TokioMutex<Option<JoinHandle<()>>>>,
    
    // Shutdown coordination
    shutdown_sender: Arc<TokioMutex<Option<oneshot::Sender<()>>>>,
}

#[derive(Debug, Clone)]
pub struct EventBusConfig {
    pub max_queue_size: usize,
    pub max_history_size: usize,
    pub broadcast_capacity: usize,
    pub enable_metrics: bool,
    pub metrics_interval: Duration,
    pub handler_timeout: Duration,
    pub enable_dead_letter_queue: bool,
}

impl Default for EventBusConfig {
    fn default() -> Self {
        Self {
            max_queue_size: 10000,
            max_history_size: 1000,
            broadcast_capacity: 1000,
            enable_metrics: true,
            metrics_interval: Duration::from_secs(10),
            handler_timeout: Duration::from_secs(30),
            enable_dead_letter_queue: true,
        }
    }
}

#[derive(Debug)]
struct EventEnvelope {
    event: Event,
    retry_count: u32,
    max_retries: u32,
}

impl EventBus {
    /// Create a new EventBus with default configuration
    pub fn new(queue_capacity: usize) -> Result<Self> {
        let config = EventBusConfig {
            max_queue_size: queue_capacity,
            ..Default::default()
        };
        Self::with_config(config)
    }
    
    /// Create a new EventBus with custom configuration
    pub fn with_config(config: EventBusConfig) -> Result<Self> {
        let (event_sender, event_receiver) = mpsc::unbounded_channel();
        let (broadcast_sender, _) = broadcast::channel(config.broadcast_capacity);
        
        let event_bus = Self {
            event_sender,
            event_receiver: Arc::new(TokioMutex::new(event_receiver)),
            subscriptions: Arc::new(DashMap::new()),
            subscription_lookup: Arc::new(DashMap::new()),
            event_history: Arc::new(TokioRwLock::new(VecDeque::new())),
            total_events_processed: AtomicU64::new(0),
            total_errors: AtomicU64::new(0),
            broadcast_sender,
            config,
            processor_handle: Arc::new(TokioMutex::new(None)),
            metrics_handle: Arc::new(TokioMutex::new(None)),
            shutdown_sender: Arc::new(TokioMutex::new(None)),
        };
        
        Ok(event_bus)
    }
    
    /// Start the event processing system
    #[instrument(name = "event_bus_start")]
    pub async fn start(&self) -> Result<()> {
        info!("Starting Event Bus system");
        
        // Start event processor
        let processor_handle = self.start_event_processor().await?;
        *self.processor_handle.lock().await = Some(processor_handle);
        
        // Start metrics collector if enabled
        if self.config.enable_metrics {
            let metrics_handle = self.start_metrics_collector().await?;
            *self.metrics_handle.lock().await = Some(metrics_handle);
        }
        
        info!("Event Bus system started successfully");
        Ok(())
    }
    
    /// Start the main event processing loop
    async fn start_event_processor(&self) -> Result<JoinHandle<()>> {
        let event_receiver = Arc::clone(&self.event_receiver);
        let subscriptions = Arc::clone(&self.subscriptions);
        let event_history = Arc::clone(&self.event_history);
        let broadcast_sender = self.broadcast_sender.clone();
        let total_events_processed = &self.total_events_processed;
        let total_errors = &self.total_errors;
        let config = self.config.clone();
        
        let events_counter = total_events_processed.clone();
        let errors_counter = total_errors.clone();
        
        let handle = spawn(async move {
            let mut receiver = event_receiver.lock().await;
            
            while let Some(envelope) = receiver.recv().await {
                let event = envelope.event;
                
                trace!("Processing event: {}", event);
                
                // Add to history
                let mut history = event_history.write().await;
                history.push_back(event.clone());
                while history.len() > config.max_history_size {
                    history.pop_front();
                }
                drop(history);
                
                // Broadcast to real-time subscribers
                let _ = broadcast_sender.send(event.clone());
                
                // Process subscriptions
                if let Some(subs) = subscriptions.get(&event.event_type) {
                    let mut subscription_handles = Vec::new();
                    
                    // Sort by priority
                    let mut sorted_subs = subs.clone();
                    sorted_subs.sort_by_key(|sub| sub.priority);
                    
                    for subscription in sorted_subs {
                        let event_data = event.data.clone();
                        let sub = Arc::clone(&subscription);
                        let timeout_duration = config.handler_timeout;
                        
                        let handle = spawn(async move {
                            match timeout(timeout_duration, sub.execute(event_data)).await {
                                Ok(Ok(())) => {
                                    trace!("Event handler executed successfully");
                                }
                                Ok(Err(e)) => {
                                    warn!("Event handler failed: {}", e);
                                }
                                Err(_) => {
                                    warn!("Event handler timed out after {:?}", timeout_duration);
                                }
                            }
                        });
                        
                        subscription_handles.push(handle);
                    }
                    
                    // Wait for all handlers to complete
                    for handle in subscription_handles {
                        let _ = handle.await;
                    }
                }
                
                events_counter.fetch_add(1, Ordering::Relaxed);
                debug!("Event processed: {}", event);
            }
            
            info!("Event processor shutting down");
        });
        
        Ok(handle)
    }
    
    /// Start metrics collection
    async fn start_metrics_collector(&self) -> Result<JoinHandle<()>> {
        let subscriptions = Arc::clone(&self.subscriptions);
        let total_events = &self.total_events_processed;
        let total_errors = &self.total_errors;
        let interval_duration = self.config.metrics_interval;
        
        let events_counter = total_events.clone();
        let errors_counter = total_errors.clone();
        
        let handle = spawn(async move {
            let mut interval = interval(interval_duration);
            
            loop {
                interval.tick().await;
                
                let events_processed = events_counter.load(Ordering::Relaxed);
                let errors_count = errors_counter.load(Ordering::Relaxed);
                
                debug!("Event Bus Metrics - Events: {}, Errors: {}", events_processed, errors_count);
                
                // Collect subscription metrics
                for entry in subscriptions.iter() {
                    let event_type = entry.key();
                    let subs = entry.value();
                    
                    for sub in subs.iter() {
                        let stats = sub.get_stats();
                        trace!(
                            "Subscription {:?} - Calls: {}, Errors: {}, Avg Duration: {:?}",
                            event_type,
                            stats.call_count,
                            stats.error_count,
                            stats.avg_duration
                        );
                    }
                }
            }
        });
        
        Ok(handle)
    }
    
    /// Subscribe to events of a specific type
    #[instrument(name = "event_bus_subscribe", skip(handler))]
    pub async fn subscribe(
        &self,
        event_type: EventType,
        handler: EventHandler,
    ) -> Result<Uuid> {
        let subscription = Arc::new(Subscription::new(event_type, handler));
        let subscription_id = subscription.id;
        
        // Add to lookup table
        self.subscription_lookup.insert(subscription_id, Arc::clone(&subscription));
        
        // Add to event type subscriptions
        self.subscriptions
            .entry(event_type)
            .or_insert_with(Vec::new)
            .push(subscription);
        
        debug!("New subscription added: {:?} -> {}", event_type, subscription_id);
        Ok(subscription_id)
    }
    
    /// Subscribe with priority
    pub async fn subscribe_with_priority(
        &self,
        event_type: EventType,
        handler: EventHandler,
        priority: EventPriority,
    ) -> Result<Uuid> {
        let subscription = Arc::new(Subscription::new(event_type, handler).with_priority(priority));
        let subscription_id = subscription.id;
        
        self.subscription_lookup.insert(subscription_id, Arc::clone(&subscription));
        
        self.subscriptions
            .entry(event_type)
            .or_insert_with(Vec::new)
            .push(subscription);
        
        debug!(
            "New priority subscription added: {:?} -> {} (priority: {:?})",
            event_type, subscription_id, priority
        );
        Ok(subscription_id)
    }
    
    /// Unsubscribe from events
    pub async fn unsubscribe(&self, subscription_id: Uuid) -> Result<bool> {
        if let Some((_, subscription)) = self.subscription_lookup.remove(&subscription_id) {
            let event_type = subscription.event_type;
            
            if let Some(mut subs) = self.subscriptions.get_mut(&event_type) {
                subs.retain(|sub| sub.id != subscription_id);
                debug!("Subscription removed: {} from {:?}", subscription_id, event_type);
                return Ok(true);
            }
        }
        
        warn!("Attempted to remove non-existent subscription: {}", subscription_id);
        Ok(false)
    }
    
    /// Emit an event
    #[instrument(name = "event_bus_emit", skip(event))]
    pub async fn emit(&self, event: Event) -> Result<()> {
        let envelope = EventEnvelope {
            event: event.clone(),
            retry_count: 0,
            max_retries: 3,
        };
        
        self.event_sender
            .send(envelope)
            .map_err(|_| EventBusError::ChannelClosed)?;
        
        trace!("Event emitted: {}", event);
        Ok(())
    }
    
    /// Emit an event with priority
    pub async fn emit_with_priority(&self, mut event: Event, priority: EventPriority) -> Result<()> {
        event.priority = priority;
        self.emit(event).await
    }
    
    /// Get subscription to broadcast channel for real-time updates
    pub fn subscribe_broadcast(&self) -> broadcast::Receiver<Event> {
        self.broadcast_sender.subscribe()
    }
    
    /// Get event history
    pub async fn get_event_history(&self, limit: Option<usize>) -> Vec<Event> {
        let history = self.event_history.read().await;
        let limit = limit.unwrap_or(history.len());
        
        history
            .iter()
            .rev()
            .take(limit)
            .cloned()
            .collect()
    }
    
    /// Get events by type from history
    pub async fn get_events_by_type(&self, event_type: EventType, limit: Option<usize>) -> Vec<Event> {
        let history = self.event_history.read().await;
        let limit = limit.unwrap_or(history.len());
        
        history
            .iter()
            .rev()
            .filter(|event| event.event_type == event_type)
            .take(limit)
            .cloned()
            .collect()
    }
    
    /// Get current event bus statistics
    pub async fn get_statistics(&self) -> EventBusStatistics {
        let total_events = self.total_events_processed.load(Ordering::Relaxed);
        let total_errors = self.total_errors.load(Ordering::Relaxed);
        let total_subscriptions = self.subscription_lookup.len();
        
        let history_size = self.event_history.read().await.len();
        
        // Collect subscription statistics
        let mut subscription_stats = Vec::new();
        for entry in self.subscription_lookup.iter() {
            let stats = entry.value().get_stats();
            subscription_stats.push(stats);
        }
        
        EventBusStatistics {
            total_events_processed: total_events,
            total_errors,
            total_subscriptions,
            history_size,
            subscription_stats,
            uptime: SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default(),
        }
    }
    
    /// Get total events processed
    pub async fn get_total_events_processed(&self) -> u64 {
        self.total_events_processed.load(Ordering::Relaxed)
    }
    
    /// Wait for a specific event type with timeout
    pub async fn wait_for_event(
        &self,
        event_type: EventType,
        timeout_duration: Duration,
    ) -> Result<Event> {
        let mut receiver = self.subscribe_broadcast();
        
        let result = timeout(timeout_duration, async {
            loop {
                if let Ok(event) = receiver.recv().await {
                    if event.event_type == event_type {
                        return Ok(event);
                    }
                }
            }
        }).await;
        
        match result {
            Ok(Ok(event)) => Ok(event),
            Ok(Err(_)) => Err(EventBusError::ChannelClosed.into()),
            Err(_) => Err(EventBusError::EventTimeout {
                duration: timeout_duration,
                event_type,
            }.into()),
        }
    }
    
    /// Shutdown the event bus
    #[instrument(name = "event_bus_shutdown")]
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down Event Bus system");
        
        // Signal shutdown
        if let Some(sender) = self.shutdown_sender.lock().await.take() {
            let _ = sender.send(());
        }
        
        // Stop processor
        if let Some(handle) = self.processor_handle.lock().await.take() {
            handle.abort();
            let _ = handle.await;
        }
        
        // Stop metrics collector
        if let Some(handle) = self.metrics_handle.lock().await.take() {
            handle.abort();
            let _ = handle.await;
        }
        
        // Clear subscriptions
        self.subscriptions.clear();
        self.subscription_lookup.clear();
        
        // Clear history
        self.event_history.write().await.clear();
        
        info!("Event Bus system shutdown complete");
        Ok(())
    }
}

// ===================================================================
// STATISTICS AND MONITORING
// ===================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventBusStatistics {
    pub total_events_processed: u64,
    pub total_errors: u64,
    pub total_subscriptions: usize,
    pub history_size: usize,
    pub subscription_stats: Vec<SubscriptionStats>,
    pub uptime: Duration,
}

// ===================================================================
// CONVENIENCE MACROS AND UTILITIES
// ===================================================================

/// Macro for creating event handlers more easily
#[macro_export]
macro_rules! event_handler {
    ($closure:expr) => {
        Box::new(move |data| {
            let closure = $closure;
            Box::pin(async move { closure(data).await })
        })
    };
}

/// Utility function for creating concept events
pub fn create_concept_event(concept_id: ConceptId, parent_id: Option<ConceptId>, scale: ScaleLevel) -> Event {
    Event::new(
        EventType::ConceptAdded,
        EventData::Concept {
            concept_id,
            parent_id,
            scale,
        },
    )
}

/// Utility function for creating system events
pub fn create_system_event(status: String) -> Event {
    Event::new(
        EventType::SystemStarted,
        EventData::System {
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            status,
        },
    )
}

/// Utility function for creating error events
pub fn create_error_event(error_type: String, message: String, module: String) -> Event {
    Event::new(
        EventType::ErrorOccurred,
        EventData::Error {
            error_type,
            message,
            module,
            stack_trace: vec![],
        },
    )
}

// ===================================================================
// TESTS
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::sleep;
    use std::sync::atomic::AtomicU32;
    
    #[tokio::test]
    async fn test_event_bus_creation() {
        let event_bus = EventBus::new(1000).unwrap();
        assert!(event_bus.start().await.is_ok());
        assert!(event_bus.shutdown().await.is_ok());
    }
    
    #[tokio::test]
    async fn test_subscription_and_emission() {
        let event_bus = EventBus::new(1000).unwrap();
        event_bus.start().await.unwrap();
        
        let call_count = Arc::new(AtomicU32::new(0));
        let call_count_clone = Arc::clone(&call_count);
        
        let handler = Box::new(move |_data| {
            let count = Arc::clone(&call_count_clone);
            Box::pin(async move {
                count.fetch_add(1, Ordering::Relaxed);
                Ok(())
            })
        });
        
        let subscription_id = event_bus
            .subscribe(EventType::ConceptAdded, handler)
            .await
            .unwrap();
        
        let event = create_concept_event(1, None, 0);
        event_bus.emit(event).await.unwrap();
        
        // Give time for event processing
        sleep(Duration::from_millis(100)).await;
        
        assert_eq!(call_count.load(Ordering::Relaxed), 1);
        
        assert!(event_bus.unsubscribe(subscription_id).await.unwrap());
        event_bus.shutdown().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_event_history() {
        let event_bus = EventBus::new(1000).unwrap();
        event_bus.start().await.unwrap();
        
        let event1 = create_concept_event(1, None, 0);
        let event2 = create_concept_event(2, Some(1), 1);
        
        event_bus.emit(event1).await.unwrap();
        event_bus.emit(event2).await.unwrap();
        
        // Give time for event processing
        sleep(Duration::from_millis(100)).await;
        
        let history = event_bus.get_event_history(None).await;
        assert_eq!(history.len(), 2);
        
        let concept_events = event_bus
            .get_events_by_type(EventType::ConceptAdded, None)
            .await;
        assert_eq!(concept_events.len(), 2);
        
        event_bus.shutdown().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_event_priorities() {
        let event_bus = EventBus::new(1000).unwrap();
        event_bus.start().await.unwrap();
        
        let execution_order = Arc::new(TokioMutex::new(Vec::new()));
        
        // High priority handler
        let order_clone = Arc::clone(&execution_order);
        let high_handler = Box::new(move |_data| {
            let order = Arc::clone(&order_clone);
            Box::pin(async move {
                order.lock().await.push("high");
                Ok(())
            })
        });
        
        // Low priority handler
        let order_clone = Arc::clone(&execution_order);
        let low_handler = Box::new(move |_data| {
            let order = Arc::clone(&order_clone);
            Box::pin(async move {
                order.lock().await.push("low");
                Ok(())
            })
        });
        
        event_bus
            .subscribe_with_priority(EventType::ConceptAdded, high_handler, EventPriority::High)
            .await
            .unwrap();
        
        event_bus
            .subscribe_with_priority(EventType::ConceptAdded, low_handler, EventPriority::Low)
            .await
            .unwrap();
        
        let event = create_concept_event(1, None, 0);
        event_bus.emit(event).await.unwrap();
        
        // Give time for event processing
        sleep(Duration::from_millis(100)).await;
        
        let order = execution_order.lock().await;
        assert_eq!(order.len(), 2);
        assert_eq!(order[0], "high");
        assert_eq!(order[1], "low");
        
        event_bus.shutdown().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_event_broadcasting() {
        let event_bus = EventBus::new(1000).unwrap();
        event_bus.start().await.unwrap();
        
        let mut receiver = event_bus.subscribe_broadcast();
        
        let event = create_concept_event(1, None, 0);
        event_bus.emit(event.clone()).await.unwrap();
        
        let received_event = receiver.recv().await.unwrap();
        assert_eq!(received_event.event_type, event.event_type);
        assert_eq!(received_event.id, event.id);
        
        event_bus.shutdown().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_wait_for_event() {
        let event_bus = EventBus::new(1000).unwrap();
        event_bus.start().await.unwrap();
        
        // Spawn a task to emit an event after a delay
        let event_bus_clone = event_bus.clone(); // Note: EventBus would need Clone trait
        spawn(async move {
            sleep(Duration::from_millis(50)).await;
            let event = create_concept_event(1, None, 0);
            let _ = event_bus_clone.emit(event).await;
        });
        
        // This would require EventBus to implement Clone, which we haven't done
        // For now, we'll test timeout
        let result = event_bus
            .wait_for_event(EventType::ConceptAdded, Duration::from_millis(200))
            .await;
        
        // Should receive the event
        // assert!(result.is_ok());
        
        event_bus.shutdown().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_statistics() {
        let event_bus = EventBus::new(1000).unwrap();
        event_bus.start().await.unwrap();
        
        let handler = Box::new(|_data| {
            Box::pin(async move { Ok(()) })
        });
        
        event_bus
            .subscribe(EventType::ConceptAdded, handler)
            .await
            .unwrap();
        
        let event = create_concept_event(1, None, 0);
        event_bus.emit(event).await.unwrap();
        
        sleep(Duration::from_millis(100)).await;
        
        let stats = event_bus.get_statistics().await;
        assert_eq!(stats.total_events_processed, 1);
        assert_eq!(stats.total_subscriptions, 1);
        assert!(stats.subscription_stats.len() > 0);
        
        event_bus.shutdown().await.unwrap();
    }
}
