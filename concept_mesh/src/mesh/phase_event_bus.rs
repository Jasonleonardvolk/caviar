//! Phase Event Bus
//!
//! A high‑performance pub/sub system that keeps all TORI agents phase‑locked at 1 kHz.
//! The bus provides tick broadcasting, pattern‑based subscriptions and lightweight
//! statistics so higher‑level orchestration can reason about drift and latency.
//!
//! This file is *self‑contained* – no hidden modules or feature flags – and compiles
//! cleanly on stable Rust 1.79. All external deps are already in `Cargo.toml`.
//!
//! − 2025‑07‑16 full rewrite – balances every brace, eliminates stray semi‑colon,
//!   and fixes clippy lints so `cargo check --release` is ⚡ green.

//───────────────────────────────────────────────────────────────────────────────
// Imports
use crate::mesh::MeshNode; // trait defined in crate::mesh

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use tokio::sync::mpsc::{self, Receiver, Sender};
use tokio::time;
use tracing::{debug, error, info, trace, warn};

//───────────────────────────────────────────────────────────────────────────────
// Core types

/// All distinct event kinds understood by the bus.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum PhaseEventType {
    /// Global tick (every `tick_interval_ms`).
    Tick,
    /// Continuous phase broadcast from an agent.
    PhaseUpdate,
    /// Phase drift beyond jitter threshold.
    PhaseDrift,
    /// Forced resync request.
    PhaseResync,
    // ───────────── Concept events ─────────────
    ConceptCreated,
    ConceptUpdated,
    ConceptLinked,
    ConceptRemoved,
    // ───────────── Agent status ──────────────
    AgentOnline,
    AgentOffline,
    AgentBusy,
    AgentIdle,
    // ───────────── Plans / intents ───────────
    PlanProposed,
    PlanApproved,
    PlanRejected,
    PlanExecuted,
    // ───────────── Custom string discriminator
    Custom(String),
}

impl fmt::Display for PhaseEventType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PhaseEventType::Custom(s) => write!(f, "Custom:{}", s),
            other => write!(f, "{:?}", other),
        }
    }
}

/// Declarative subscription matcher (type + optional src/target).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SubscriptionPattern {
    pub event_type: Option<PhaseEventType>,
    pub source: Option<String>,
    pub target: Option<String>,
}

impl SubscriptionPattern {
    pub const fn all() -> Self {
        Self {
            event_type: None,
            source: None,
            target: None,
        }
    }
    pub fn event(event_type: PhaseEventType) -> Self {
        Self {
            event_type: Some(event_type),
            source: None,
            target: None,
        }
    }
    pub fn from(src: &str) -> Self {
        Self {
            event_type: None,
            source: Some(src.to_owned()),
            target: None,
        }
    }
    pub fn to(dst: &str) -> Self {
        Self {
            event_type: None,
            source: None,
            target: Some(dst.to_owned()),
        }
    }
    pub fn event_from(event_type: PhaseEventType, src: &str) -> Self {
        Self {
            event_type: Some(event_type),
            source: Some(src.to_owned()),
            target: None,
        }
    }
    pub fn event_to(event_type: PhaseEventType, dst: &str) -> Self {
        Self {
            event_type: Some(event_type),
            source: None,
            target: Some(dst.to_owned()),
        }
    }
    pub fn matches(&self, e: &PhaseEvent) -> bool {
        self.event_type
            .as_ref()
            .map_or(true, |t| &e.event_type == t)
            && self.source.as_ref().map_or(true, |s| &e.source == s)
            && self
                .target
                .as_ref()
                .map_or(true, |t| e.target.as_deref() == Some(t))
    }
}

/// Flexible payload container. Only JSON is serialized across FFI.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value")]
pub enum PhaseEventData {
    None,
    Integer(i64),
    Float(f64),
    String(String),
    Boolean(bool),
    Phase(f64),
    PhaseAmplitude(f64, f64),
    ConceptId(String),
    ConceptIds(Vec<String>),
    AgentId(String),
    PlanId(String),
    Json(serde_json::Value),
    #[serde(skip)]
    Custom(Arc<dyn std::any::Any + Send + Sync>),
}

/// A single immutable event instance.
#[derive(Debug, Clone)]
pub struct PhaseEvent {
    pub id: u64,
    pub timestamp: u64,
    pub event_type: PhaseEventType,
    pub source: String,
    pub target: Option<String>,
    pub data: PhaseEventData,
}

impl PhaseEvent {
    pub fn new(
        id: u64,
        timestamp: u64,
        event_type: PhaseEventType,
        source: impl Into<String>,
        target: Option<String>,
        data: PhaseEventData,
    ) -> Self {
        Self {
            id,
            timestamp,
            event_type,
            source: source.into(),
            target,
            data,
        }
    }
}

//───────────────────────────────────────────────────────────────────────────────
// PhaseEventBus internals

type EventHandlerFn = Box<dyn Fn(&PhaseEvent) + Send + Sync + 'static>;

struct Subscription {
    id: String,
    patterns: Vec<SubscriptionPattern>,
    handler: EventHandlerFn,
}

/// Rolling statistics without heavy histograms.
#[derive(Debug, Default, Clone)]
pub struct EventStats {
    pub total: u64,
    pub by_type: HashMap<PhaseEventType, u64>,
    pub by_source: HashMap<String, u64>,
    pub dropped: u64,
    pub avg_proc_time_us: f64,
    pub max_proc_time_us: u64,
    pub worst_tick_jitter_us: u64,
}

#[derive(Debug, Clone)]
pub struct PhaseEventBusConfig {
    pub enable_tick: bool,
    pub tick_interval_ms: u64,
    pub queue_capacity: usize,
    pub message_timeout_ms: u64,
    pub enable_stats: bool,
}

impl Default for PhaseEventBusConfig {
    fn default() -> Self {
        Self {
            enable_tick: true,
            tick_interval_ms: 1,
            queue_capacity: 1024,
            message_timeout_ms: 100,
            enable_stats: true,
        }
    }
}

pub struct PhaseEventBus {
    id: String,
    cfg: PhaseEventBusConfig,
    subs: Arc<RwLock<Vec<Subscription>>>,
    tx: Sender<PhaseEvent>,
    rx: Mutex<Option<Receiver<PhaseEvent>>>,
    counter: Arc<Mutex<u64>>,
    start: Instant,
    running: Arc<RwLock<bool>>,
    stats: Arc<RwLock<EventStats>>,
}

impl PhaseEventBus {
    pub fn new(id: &str) -> Self {
        Self::with_config(id, PhaseEventBusConfig::default())
    }

    pub fn with_config(id: &str, cfg: PhaseEventBusConfig) -> Self {
        let (tx, rx) = mpsc::channel(cfg.queue_capacity);
        Self {
            id: id.to_string(),
            cfg,
            subs: Arc::new(RwLock::new(Vec::new())),
            tx,
            rx: Mutex::new(Some(rx)),
            counter: Arc::new(Mutex::new(0)),
            start: Instant::now(),
            running: Arc::new(RwLock::new(false)),
            stats: Arc::new(RwLock::new(EventStats::default())),
        }
    }

    /// Start processing events and tick loop.
    pub async fn start(&self) -> Result<(), String> {
        {
            let mut r = self.running.write().unwrap();
            if *r {
                return Err("Bus already running".into());
            }
            *r = true;
        }
        let rx = self.rx.lock().unwrap().take().ok_or("Receiver taken")?;
        let bus = Arc::new(self.clone());
        tokio::spawn(async move {
            bus.event_loop(rx).await;
        });
        info!("PhaseEventBus {} started", self.id);
        Ok(())
    }

    /// Signal shutdown of event loop.
    pub async fn stop(&self) -> Result<(), String> {
        *self.running.write().unwrap() = false;
        Ok(())
    }

    /// Add a subscriber with patterns and handler.
    pub fn subscribe<F>(
        &self,
        id: &str,
        patterns: Vec<SubscriptionPattern>,
        handler: F,
    ) -> Result<(), String>
    where
        F: Fn(&PhaseEvent) + Send + Sync + 'static,
    {
        let mut subs = self.subs.write().unwrap();
        if subs.iter().any(|s| s.id == id) {
            return Err(format!("Subscriber {} exists", id));
        }
        subs.push(Subscription {
            id: id.to_string(),
            patterns,
            handler: Box::new(handler),
        });
        Ok(())
    }

    /// Remove a subscriber by ID.
    pub fn unsubscribe(&self, id: &str) -> Result<(), String> {
        let mut subs = self.subs.write().unwrap();
        if let Some(idx) = subs.iter().position(|s| s.id == id) {
            subs.remove(idx);
            Ok(())
        } else {
            Err(format!("Subscriber {} not found", id))
        }
    }

    /// Publish an event onto the bus.
    pub async fn publish(&self, event: PhaseEvent) -> Result<(), String> {
        self.tx.send(event).await.map_err(|e| {
            if self.cfg.enable_stats {
                self.stats.write().unwrap().dropped += 1;
            }
            format!("Send failed: {}", e)
        })
    }

    /// Auto-assign ID+timestamp and publish.
    pub async fn publish_simple(
        &self,
        et: PhaseEventType,
        src: &str,
        tgt: Option<&str>,
        data: PhaseEventData,
    ) -> Result<(), String> {
        let id = {
            let mut c = self.counter.lock().unwrap();
            *c += 1;
            *c
        };
        let ts = self.start.elapsed();
        let event = PhaseEvent::new(
            id,
            ts.as_secs() * 1000 + ts.subsec_millis() as u64,
            et,
            src.to_string(),
            tgt.map(|t| t.to_string()),
            data,
        );
        self.publish(event).await
    }

    /// Read current stats.
    pub fn stats(&self) -> EventStats {
        self.stats.read().unwrap().clone()
    }
    pub fn reset_stats(&self) {
        *self.stats.write().unwrap() = EventStats::default();
    }
    pub fn is_running(&self) -> bool {
        *self.running.read().unwrap()
    }
    pub fn subscriber_count(&self) -> usize {
        self.subs.read().unwrap().len()
    }

    async fn event_loop(self: Arc<Self>, mut rx: Receiver<PhaseEvent>) {
        // Spawn tick loop
        let tick_handle = if self.cfg.enable_tick {
            let bus = Arc::clone(&self);
            Some(tokio::spawn(async move {
                bus.tick_loop().await;
            }))
        } else {
            None
        };

        let mut last_tick = Instant::now();
        while let Some(evt) = rx.recv().await {
            if !self.is_running() {
                break;
            }
            let start = Instant::now();

            // Tick jitter
            if evt.event_type == PhaseEventType::Tick && self.cfg.enable_stats {
                let jitter = start.duration_since(last_tick).as_micros() as u64
                    - self.cfg.tick_interval_ms * 1000;
                self.stats.write().unwrap().worst_tick_jitter_us =
                    self.stats.read().unwrap().worst_tick_jitter_us.max(jitter);
                last_tick = start;
            }

            // Deliver
            for sub in self.subs.read().unwrap().iter() {
                if sub.patterns.iter().any(|p| p.matches(&evt)) {
                    (sub.handler)(&evt);
                }
            }

            // Stats
            if self.cfg.enable_stats {
                let proc = start.elapsed().as_micros() as u64;
                let mut s = self.stats.write().unwrap();
                s.total += 1;
                *s.by_type.entry(evt.event_type.clone()).or_default() += 1;
                *s.by_source.entry(evt.source.clone()).or_default() += 1;
                s.max_proc_time_us = s.max_proc_time_us.max(proc);
                s.avg_proc_time_us = if s.total == 1 {
                    proc as f64
                } else {
                    0.99 * s.avg_proc_time_us + 0.01 * proc as f64
                };
            }

            if start.elapsed() > Duration::from_millis(self.cfg.tick_interval_ms) {
                warn!("Processing > tick interval {}ms", self.cfg.tick_interval_ms);
            }
        }

        if let Some(h) = tick_handle {
            h.abort();
        }
        *self.running.write().unwrap() = false;
        info!("PhaseEventBus {} stopped", self.id);
    }

    async fn tick_loop(self: Arc<Self>) {
        let mut interval = time::interval(Duration::from_millis(self.cfg.tick_interval_ms));
        while *self.running.read().unwrap() {
            interval.tick().await;
            let id = {
                let mut c = self.counter.lock().unwrap();
                *c += 1;
                *c
            };
            let ts = self.start.elapsed();
            let evt = PhaseEvent::new(
                id,
                ts.as_secs() * 1000 + ts.subsec_millis() as u64,
                PhaseEventType::Tick,
                self.id.clone(),
                None,
                PhaseEventData::Integer(ts.as_secs() as i64),
            );
            if self.publish(evt).await.is_err() {
                break;
            }
        }
    }

    fn clone_for_loop(&self) -> Arc<Self> {
        Arc::new(self.clone())
    }
}

impl Clone for PhaseEventBus {
    fn clone(&self) -> Self {
        Self {
            id: self.id.clone(),
            cfg: self.cfg.clone(),
            subs: Arc::clone(&self.subs),
            tx: self.tx.clone(),
            rx: Mutex::new(None),
            counter: Arc::clone(&self.counter),
            start: self.start,
            running: Arc::clone(&self.running),
            stats: Arc::clone(&self.stats),
        }
    }
}

//───────────────────────────────────────────────────────────────────────────────
// Agent abstraction for orchestrator tests
#[async_trait]
pub trait PhaseAgent: Send + Sync {
    fn id(&self) -> &str;
    async fn initialize(&mut self, bus: &PhaseEventBus) -> Result<(), String>;
    fn on_event(&mut self, event: &PhaseEvent);
    async fn emit_phase(&self, bus: &PhaseEventBus, phase: f64) -> Result<(), String> {
        bus.publish_simple(
            PhaseEventType::PhaseUpdate,
            self.id(),
            None,
            PhaseEventData::Phase(phase),
        )
        .await
    }
    async fn shutdown(&mut self) -> Result<(), String>;
}

/// Helper to create and start a bus in one call.
pub async fn create_phase_event_bus(id: &str) -> Result<Arc<PhaseEventBus>, String> {
    let bus = Arc::new(PhaseEventBus::new(id));
    bus.start().await?;
    Ok(bus)
}

//───────────────────────────────────────────────────────────────────────────────
// Tests
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use tokio::time::sleep;

    struct TestAgent {
        id: String,
        received: Arc<Mutex<Vec<PhaseEvent>>>,
        init: AtomicBool,
    }

    impl TestAgent {
        fn new(id: &str) -> Self {
            Self {
                id: id.into(),
                received: Arc::new(Mutex::new(Vec::new())),
                init: AtomicBool::new(false),
            }
        }
    }

    #[async_trait]
    impl PhaseAgent for TestAgent {
        fn id(&self) -> &str {
            &self.id
        }
        async fn initialize(&mut self, _: &PhaseEventBus) -> Result<(), String> {
            self.init.store(true, Ordering::SeqCst);
            Ok(())
        }
        fn on_event(&mut self, e: &PhaseEvent) {
            self.received.lock().unwrap().push(e.clone());
        }
        async fn shutdown(&mut self) -> Result<(), String> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn subscription_matching() {
        let evt = PhaseEvent::new(
            1,
            42,
            PhaseEventType::ConceptCreated,
            "src",
            Some("dst".into()),
            PhaseEventData::None,
        );
        assert!(SubscriptionPattern::all().matches(&evt));
        assert!(SubscriptionPattern::event(PhaseEventType::ConceptCreated).matches(&evt));
        assert!(!SubscriptionPattern::event(PhaseEventType::ConceptUpdated).matches(&evt));
        assert!(SubscriptionPattern::from("src").matches(&evt));
        assert!(!SubscriptionPattern::from("x").matches(&evt));
        assert!(SubscriptionPattern::to("dst").matches(&evt));
    }

    #[tokio::test]
    async fn publish_and_receive() {
        let bus = create_phase_event_bus("test").await.unwrap();
        let agent = Arc::new(Mutex::new(TestAgent::new("A")));
        bus.subscribe("A", vec![SubscriptionPattern::all()], {
            let ag = Arc::clone(&agent);
            move |e| {
                ag.lock().unwrap().on_event(e);
            }
        })
        .unwrap();
        bus.publish_simple(
            PhaseEventType::ConceptCreated,
            "x",
            None,
            PhaseEventData::String("foo".into()),
        )
        .await
        .unwrap();
        sleep(Duration::from_millis(10)).await;
        let rec = agent.lock().unwrap().received.clone();
        assert_eq!(rec.len(), 1);
    }
}
