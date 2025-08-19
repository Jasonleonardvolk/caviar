//! Phase Event Bus Demo
//!
//! This is a demonstration of the Phase Event Bus, showing how multiple agents
//! can communicate and maintain phase coherence at 1kHz. The demo creates several
//! agents that emit phase updates and synchronize with each other.

use concept_mesh_rs::mesh::{
    create_phase_event_bus, PhaseAgent, PhaseEvent, PhaseEventBus, PhaseEventData, PhaseEventType,
    SubscriptionPattern,
};

use async_trait::async_trait;
use clap::Parser;
use std::collections::{HashMap, VecDeque};
use std::f64::consts::PI;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio::time;
use tracing::{debug, error, info, warn};

/// Command line arguments
#[derive(Parser, Debug)]
#[clap(name = "phase-bus-demo", about = "Demonstrate the Phase Event Bus")]
struct Args {
    /// Number of agents to create
    #[clap(short, long, default_value = "3")]
    agents: usize,

    /// Running time in seconds
    #[clap(short, long, default_value = "10")]
    time: u64,

    /// Verbosity level (0-3)
    #[clap(short, long, default_value = "1")]
    verbose: u8,

    /// Enable phase drift simulation
    #[clap(long)]
    simulate_drift: bool,

    /// Enable perturbation simulation
    #[clap(long)]
    simulate_perturbation: bool,

    /// Print statistics every N seconds
    #[clap(short, long, default_value = "1")]
    stats_interval: u64,
}

/// Demo agent that participates in the phase event bus
struct DemoAgent {
    /// Agent ID
    id: String,

    /// Natural frequency (radians/second)
    natural_frequency: f64,

    /// Current phase (0.0 - 1.0)
    phase: Arc<RwLock<f64>>,

    /// Phase history (for visualization)
    phase_history: Arc<Mutex<VecDeque<(u64, f64)>>>,

    /// Other agents' phases
    other_phases: Arc<RwLock<HashMap<String, f64>>>,

    /// Coupling strength
    coupling_strength: f64,

    /// Running flag
    running: Arc<RwLock<bool>>,

    /// Last update time
    last_update: Arc<Mutex<Instant>>,

    /// Event bus sender
    event_sender: Arc<Mutex<Option<mpsc::Sender<PhaseEvent>>>>,

    /// Statistics
    stats: Arc<Mutex<AgentStats>>,
}

/// Agent statistics
#[derive(Debug, Default, Clone)]
struct AgentStats {
    /// Total events received
    events_received: u64,

    /// Phase updates received
    phase_updates: u64,

    /// Phase updates sent
    phase_updates_sent: u64,

    /// Average phase difference from group
    avg_phase_diff: f64,

    /// Maximum phase difference
    max_phase_diff: f64,

    /// Phase velocity (change per second)
    phase_velocity: f64,
}

impl DemoAgent {
    /// Create a new demo agent
    fn new(id: &str, natural_frequency: f64, coupling_strength: f64) -> Self {
        Self {
            id: id.to_string(),
            natural_frequency,
            phase: Arc::new(RwLock::new(0.0)),
            phase_history: Arc::new(Mutex::new(VecDeque::with_capacity(1000))),
            other_phases: Arc::new(RwLock::new(HashMap::new())),
            coupling_strength,
            running: Arc::new(RwLock::new(false)),
            last_update: Arc::new(Mutex::new(Instant::now())),
            event_sender: Arc::new(Mutex::new(None)),
            stats: Arc::new(Mutex::new(AgentStats::default())),
        }
    }

    /// Start the agent's update loop
    async fn start_update_loop(&self) -> Result<(), String> {
        let running = Arc::clone(&self.running);
        let phase = Arc::clone(&self.phase);
        let other_phases = Arc::clone(&self.other_phases);
        let last_update = Arc::clone(&self.last_update);
        let event_sender = Arc::clone(&self.event_sender);
        let phase_history = Arc::clone(&self.phase_history);
        let stats = Arc::clone(&self.stats);
        let id = self.id.clone();
        let natural_frequency = self.natural_frequency;
        let coupling_strength = self.coupling_strength;

        tokio::spawn(async move {
            // Create a 1ms interval timer
            let mut interval = time::interval(Duration::from_millis(1));

            while *running.read().unwrap() {
                interval.tick().await;

                // Calculate time delta
                let now = Instant::now();
                let dt = {
                    let mut last = last_update.lock().unwrap();
                    let dt = now.duration_since(*last).as_secs_f64();
                    *last = now;
                    dt
                };

                // Get current phase
                let current_phase = {
                    let phase = phase.read().unwrap();
                    *phase
                };

                // Calculate phase update based on natural frequency
                let mut new_phase = current_phase + natural_frequency * dt;

                // Apply phase coupling
                let phase_diff_sum = {
                    let other_phases = other_phases.read().unwrap();
                    let mut sum = 0.0;
                    let mut count = 0;

                    for (_, &other_phase) in other_phases.iter() {
                        // Calculate phase difference
                        let diff = normalize_phase_diff(other_phase - current_phase);
                        sum += diff;
                        count += 1;
                    }

                    // Calculate average phase difference
                    if count > 0 {
                        sum / count as f64
                    } else {
                        0.0
                    }
                };

                // Apply coupling force
                new_phase += coupling_strength * phase_diff_sum * dt;

                // Normalize phase to [0, 1)
                new_phase = normalize_phase(new_phase);

                // Update phase
                {
                    let mut phase = phase.write().unwrap();
                    *phase = new_phase;
                }

                // Update phase history
                {
                    let mut history = phase_history.lock().unwrap();
                    let timestamp = chrono::Utc::now().timestamp_millis() as u64;
                    history.push_back((timestamp, new_phase));

                    // Keep only the last 1000 points
                    while history.len() > 1000 {
                        history.pop_front();
                    }
                }

                // Update statistics
                {
                    let mut stats = stats.lock().unwrap();
                    stats.phase_velocity = natural_frequency + coupling_strength * phase_diff_sum;

                    // Calculate phase difference stats
                    let other_phases = other_phases.read().unwrap();
                    if !other_phases.is_empty() {
                        let mut sum_diff = 0.0;
                        let mut max_diff = 0.0;

                        for (_, &other_phase) in other_phases.iter() {
                            let diff = normalize_phase_diff(other_phase - new_phase).abs();
                            sum_diff += diff;
                            max_diff = max_diff.max(diff);
                        }

                        stats.avg_phase_diff = sum_diff / other_phases.len() as f64;
                        stats.max_phase_diff = max_diff;
                    }
                }

                // Emit phase update event
                if let Some(sender) = &*event_sender.lock().unwrap() {
                    let timestamp = chrono::Utc::now().timestamp_millis() as u64;
                    let event = PhaseEvent::phase_update(timestamp, &id, new_phase);

                    if let Err(e) = sender.send(event).await {
                        error!("Failed to send phase update: {}", e);
                    } else {
                        // Update statistics
                        let mut stats = stats.lock().unwrap();
                        stats.phase_updates_sent += 1;
                    }
                }
            }
        });

        Ok(())
    }

    /// Get the agent's statistics
    fn get_stats(&self) -> AgentStats {
        let stats = self.stats.lock().unwrap();
        stats.clone()
    }

    /// Get the agent's phase history
    fn get_phase_history(&self) -> Vec<(u64, f64)> {
        let history = self.phase_history.lock().unwrap();
        history.iter().cloned().collect()
    }
}

/// Normalize phase to [0, 1)
fn normalize_phase(phase: f64) -> f64 {
    phase.rem_euclid(1.0)
}

/// Normalize phase difference to [-0.5, 0.5)
fn normalize_phase_diff(diff: f64) -> f64 {
    let diff = diff.rem_euclid(1.0);
    if diff >= 0.5 {
        diff - 1.0
    } else {
        diff
    }
}

#[async_trait]
impl PhaseAgent for DemoAgent {
    fn id(&self) -> &str {
        &self.id
    }

    async fn initialize(&mut self, bus: &PhaseEventBus) -> Result<(), String> {
        // Store the event sender
        {
            let mut sender = self.event_sender.lock().unwrap();
            *sender = Some(bus.get_sender());
        }

        // Mark as running
        {
            let mut running = self.running.write().unwrap();
            *running = true;
        }

        // Start the update loop
        self.start_update_loop().await?;

        Ok(())
    }

    fn on_event(&mut self, event: &PhaseEvent) {
        // Update statistics
        {
            let mut stats = self.stats.lock().unwrap();
            stats.events_received += 1;
        }

        // Process different event types
        match event.event_type {
            PhaseEventType::PhaseUpdate => {
                // Ignore own phase updates
                if event.source == self.id {
                    return;
                }

                // Extract phase from event
                if let PhaseEventData::Phase(phase) = event.data {
                    // Store the other agent's phase
                    let mut other_phases = self.other_phases.write().unwrap();
                    other_phases.insert(event.source.clone(), phase);

                    // Update statistics
                    let mut stats = self.stats.lock().unwrap();
                    stats.phase_updates += 1;
                }
            }
            _ => {
                // Ignore other event types for this demo
            }
        }
    }

    async fn shutdown(&mut self) -> Result<(), String> {
        // Mark as not running
        {
            let mut running = self.running.write().unwrap();
            *running = false;
        }

        // Wait a bit for the update loop to stop
        tokio::time::sleep(Duration::from_millis(10)).await;

        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<(), String> {
    // Parse command line arguments
    let args = Args::parse();

    // Set up logging
    let log_level = match args.verbose {
        0 => tracing::Level::ERROR,
        1 => tracing::Level::INFO,
        2 => tracing::Level::DEBUG,
        _ => tracing::Level::TRACE,
    };

    tracing_subscriber::fmt::fmt()
        .with_max_level(log_level)
        .init();

    info!("Phase Event Bus Demo");
    info!("Number of agents: {}", args.agents);
    info!("Running time: {} seconds", args.time);

    // Create phase event bus
    info!("Creating Phase Event Bus...");
    let bus = create_phase_event_bus("demo").await?;

    // Create agents
    info!("Creating agents...");
    let mut agents = Vec::new();

    for i in 0..args.agents {
        // Calculate natural frequency (slightly different for each agent)
        let base_freq = 0.01; // 0.01 revolutions per second (very slow for demo purposes)
        let freq_variation = if args.simulate_drift {
            // Simulate drift with larger frequency differences
            (i as f64 * 0.002) - (0.002 * args.agents as f64 / 2.0)
        } else {
            // Small random variation
            (i as f64 * 0.0001) - (0.0001 * args.agents as f64 / 2.0)
        };

        let natural_frequency = base_freq + freq_variation;

        // Create agent
        let mut agent = DemoAgent::new(
            &format!("agent_{}", i),
            natural_frequency,
            0.1, // Coupling strength
        );

        // Subscribe to phase updates
        let agent_clone = Arc::new(Mutex::new(agent.clone()));
        bus.subscribe(
            &format!("agent_{}", i),
            vec![SubscriptionPattern::event(PhaseEventType::PhaseUpdate)],
            move |event| {
                let mut agent = agent_clone.lock().unwrap();
                agent.on_event(event);
            },
        )?;

        // Initialize the agent
        agent.initialize(&bus).await?;

        // Add to agent list
        agents.push(agent);
    }

    // Run for the specified time
    info!("Running for {} seconds...", args.time);
    let start_time = Instant::now();
    let mut next_stats_time = Instant::now() + Duration::from_secs(args.stats_interval);

    let mut perturbation_applied = false;

    while start_time.elapsed() < Duration::from_secs(args.time) {
        // Sleep a bit
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Apply perturbation if enabled
        if args.simulate_perturbation
            && !perturbation_applied
            && start_time.elapsed() > Duration::from_secs(args.time / 2)
        {
            info!("Applying perturbation to agent_0...");

            // Set the first agent's phase to a random value
            if let Some(agent) = agents.first_mut() {
                let mut phase = agent.phase.write().unwrap();
                *phase = 0.75; // Significantly out of phase
                perturbation_applied = true;
            }
        }

        // Print statistics at intervals
        if Instant::now() >= next_stats_time {
            // Time elapsed
            let elapsed = start_time.elapsed().as_secs_f64();
            info!("Time elapsed: {:.1} seconds", elapsed);

            // Bus stats
            let bus_stats = bus.get_stats();
            info!("Event bus stats:");
            info!("  Total events: {}", bus_stats.total_events);
            info!(
                "  Avg. processing time: {:.3} Î¼s",
                bus_stats.avg_processing_time_us
            );
            info!(
                "  Max processing time: {} Î¼s",
                bus_stats.max_processing_time_us
            );
            info!("  Tick jitter: {} Î¼s", bus_stats.tick_jitter_us);

            // Agent stats
            info!("Agent stats:");
            for (i, agent) in agents.iter().enumerate() {
                let stats = agent.get_stats();
                let phase = *agent.phase.read().unwrap();

                info!(
                    "  Agent {}: phase={:.3}, velocity={:.6}, avg_diff={:.3}, max_diff={:.3}, events={}",
                    i,
                    phase,
                    stats.phase_velocity,
                    stats.avg_phase_diff,
                    stats.max_phase_diff,
                    stats.events_received
                );
            }

            // Calculate group synchrony
            let mut sum_diff = 0.0;
            let mut count = 0;

            for i in 0..agents.len() {
                for j in i + 1..agents.len() {
                    let phase_i = *agents[i].phase.read().unwrap();
                    let phase_j = *agents[j].phase.read().unwrap();
                    let diff = normalize_phase_diff(phase_i - phase_j).abs();
                    sum_diff += diff;
                    count += 1;
                }
            }

            let avg_group_diff = if count > 0 {
                sum_diff / count as f64
            } else {
                0.0
            };
            info!("Group synchrony: {:.6} (lower is better)", avg_group_diff);

            // Update next stats time
            next_stats_time = Instant::now() + Duration::from_secs(args.stats_interval);
        }
    }

    // Shutdown agents
    info!("Shutting down agents...");
    for agent in &mut agents {
        agent.shutdown().await?;
    }

    // Stop the event bus
    info!("Stopping event bus...");
    bus.stop().await?;

    info!("Demo completed successfully");
    Ok(())
}

