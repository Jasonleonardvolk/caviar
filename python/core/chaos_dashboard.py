#!/usr/bin/env python3
"""
TORI Chaos Dynamics Visualization Dashboard
Real-time monitoring of chaos-enhanced cognitive processing
"""

import asyncio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle
import matplotlib.gridspec as gridspec
from collections import deque
from datetime import datetime, timezone
import json

# Import TORI components
from python.core.tori_production import TORIProductionSystem, TORIProductionConfig
from python.core.metacognitive_adapters import AdapterMode

class ChaosDynamicsDashboard:
    """Real-time visualization of TORI's chaos dynamics"""
    
    def __init__(self, tori_system: TORIProductionSystem):
        self.tori = tori_system
        
        # Data buffers
        self.eigenvalue_history = deque(maxlen=100)
        self.energy_history = deque(maxlen=100)
        self.safety_history = deque(maxlen=100)
        self.efficiency_history = deque(maxlen=100)
        self.phase_space_trajectory = deque(maxlen=500)
        
        # Timing
        self.timestamps = deque(maxlen=100)
        self.start_time = datetime.now(timezone.utc)
        
        # Setup figure
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('TORI Chaos Dynamics Monitor', fontsize=16, fontweight='bold')
        
        # Create grid layout
        gs = gridspec.GridSpec(3, 3, figure=self.fig, hspace=0.3, wspace=0.3)
        
        # Subplots
        self.ax_eigenvalues = self.fig.add_subplot(gs[0, 0])
        self.ax_phase_space = self.fig.add_subplot(gs[0, 1])
        self.ax_energy = self.fig.add_subplot(gs[0, 2])
        self.ax_safety = self.fig.add_subplot(gs[1, 0])
        self.ax_efficiency = self.fig.add_subplot(gs[1, 1])
        self.ax_chaos_modes = self.fig.add_subplot(gs[1, 2])
        self.ax_soliton = self.fig.add_subplot(gs[2, 0])
        self.ax_attractor = self.fig.add_subplot(gs[2, 1])
        self.ax_status = self.fig.add_subplot(gs[2, 2])
        
        # Initialize plots
        self._init_plots()
        
        # Animation
        self.animation = None
        
    def _init_plots(self):
        """Initialize all subplot configurations"""
        
        # Eigenvalue Monitor
        self.ax_eigenvalues.set_title('Eigenvalue Spectrum')
        self.ax_eigenvalues.set_xlabel('Time (s)')
        self.ax_eigenvalues.set_ylabel('Max Eigenvalue')
        self.ax_eigenvalues.axhline(y=1.0, color='g', linestyle='--', label='Stable')
        self.ax_eigenvalues.axhline(y=1.3, color='y', linestyle='--', label='Soft Margin')
        self.ax_eigenvalues.axhline(y=2.0, color='r', linestyle='--', label='Emergency')
        self.ax_eigenvalues.legend(loc='upper right')
        self.ax_eigenvalues.set_ylim(0, 2.5)
        
        # Phase Space
        self.ax_phase_space.set_title('Phase Space Trajectory')
        self.ax_phase_space.set_xlabel('Dimension 1')
        self.ax_phase_space.set_ylabel('Dimension 2')
        self.ax_phase_space.set_xlim(-5, 5)
        self.ax_phase_space.set_ylim(-5, 5)
        
        # Energy Flow
        self.ax_energy.set_title('Energy Distribution')
        self.ax_energy.set_xlabel('Time (s)')
        self.ax_energy.set_ylabel('Energy Units')
        
        # Safety Level
        self.ax_safety.set_title('Safety Metrics')
        self.ax_safety.set_xlabel('Time (s)')
        self.ax_safety.set_ylabel('Safety Score')
        self.ax_safety.set_ylim(0, 1.1)
        
        # Efficiency
        self.ax_efficiency.set_title('Chaos Efficiency Gain')
        self.ax_efficiency.set_xlabel('Time (s)')
        self.ax_efficiency.set_ylabel('Efficiency Multiplier')
        self.ax_efficiency.set_ylim(0, 20)
        
        # Chaos Mode Distribution
        self.ax_chaos_modes.set_title('Active Chaos Modes')
        self.chaos_mode_bars = None
        
        # Soliton Visualization
        self.ax_soliton.set_title('Dark Soliton Profile')
        self.ax_soliton.set_xlabel('Position')
        self.ax_soliton.set_ylabel('Amplitude')
        self.ax_soliton.set_ylim(0, 1.5)
        
        # Attractor Basin
        self.ax_attractor.set_title('Attractor Landscape')
        self.ax_attractor.set_xlabel('X')
        self.ax_attractor.set_ylabel('Y')
        
        # Status Text
        self.ax_status.set_title('System Status')
        self.ax_status.axis('off')
        self.status_text = self.ax_status.text(0.05, 0.95, '', 
                                              transform=self.ax_status.transAxes,
                                              verticalalignment='top',
                                              fontfamily='monospace')
        
    async def collect_data(self):
        """Collect data from TORI system"""
        # Get current time
        current_time = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        self.timestamps.append(current_time)
        
        # Get system status
        status = self.tori.get_status()
        eigen_status = status['eigensentry']
        ccl_status = status['ccl']
        safety_status = status['safety']
        
        # Collect eigenvalues
        max_eigenvalue = eigen_status['current_max_eigenvalue']
        self.eigenvalue_history.append(max_eigenvalue)
        
        # Collect energy data
        total_energy = eigen_status['energy_stats']['total_available']
        self.energy_history.append(total_energy)
        
        # Collect safety metrics
        if safety_status['metrics']:
            safety_score = (
                safety_status['metrics']['fidelity'] * 0.3 +
                safety_status['metrics']['coherence'] * 0.2 +
                safety_status['metrics']['energy_conservation'] * 0.2 +
                safety_status['metrics']['chaos_containment'] * 0.3
            )
            self.safety_history.append(safety_score)
        
        # Collect efficiency
        efficiency_ratio = ccl_status['efficiency_ratio']
        self.efficiency_history.append(efficiency_ratio)
        
        # Collect phase space data (simulated from state)
        state = self.tori.state_manager.get_state()
        if len(state) >= 2:
            self.phase_space_trajectory.append([state[0], state[1]])
            
    def update_plots(self, frame):
        """Update all plots with latest data"""
        
        # Run async data collection
        asyncio.create_task(self.collect_data())
        
        # Clear dynamic plots
        self.ax_eigenvalues.lines = self.ax_eigenvalues.lines[:3]  # Keep reference lines
        self.ax_energy.clear()
        self.ax_safety.clear()
        self.ax_efficiency.clear()
        
        # Re-init cleared plots
        self.ax_energy.set_title('Energy Distribution')
        self.ax_energy.set_xlabel('Time (s)')
        self.ax_energy.set_ylabel('Energy Units')
        
        self.ax_safety.set_title('Safety Metrics')
        self.ax_safety.set_xlabel('Time (s)')
        self.ax_safety.set_ylabel('Safety Score')
        self.ax_safety.set_ylim(0, 1.1)
        
        self.ax_efficiency.set_title('Chaos Efficiency Gain')
        self.ax_efficiency.set_xlabel('Time (s)')
        self.ax_efficiency.set_ylabel('Efficiency Multiplier')
        self.ax_efficiency.set_ylim(0, 20)
        
        if len(self.timestamps) > 0:
            # Update eigenvalue plot
            self.ax_eigenvalues.plot(self.timestamps, self.eigenvalue_history, 'b-', linewidth=2)
            
            # Update energy plot
            self.ax_energy.plot(self.timestamps, self.energy_history, 'g-', linewidth=2)
            self.ax_energy.fill_between(self.timestamps, 0, self.energy_history, alpha=0.3)
            
            # Update safety plot
            if self.safety_history:
                self.ax_safety.plot(self.timestamps, self.safety_history, 'purple', linewidth=2)
                self.ax_safety.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5)
                self.ax_safety.axhline(y=0.3, color='red', linestyle='--', alpha=0.5)
            
            # Update efficiency plot
            self.ax_efficiency.plot(self.timestamps, self.efficiency_history, 'orange', linewidth=2)
            self.ax_efficiency.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
            
            # Add efficiency gain regions
            for i, eff in enumerate(self.efficiency_history):
                if eff > 10:
                    self.ax_efficiency.axvspan(self.timestamps[i]-0.1, self.timestamps[i]+0.1,
                                             alpha=0.2, color='red', label='Phase Explosion' if i == 0 else '')
                elif eff > 5:
                    self.ax_efficiency.axvspan(self.timestamps[i]-0.1, self.timestamps[i]+0.1,
                                             alpha=0.2, color='orange', label='Attractor Hop' if i == 0 else '')
                elif eff > 3:
                    self.ax_efficiency.axvspan(self.timestamps[i]-0.1, self.timestamps[i]+0.1,
                                             alpha=0.2, color='yellow', label='Dark Soliton' if i == 0 else '')
        
        # Update phase space
        if self.phase_space_trajectory:
            trajectory = np.array(list(self.phase_space_trajectory))
            self.ax_phase_space.clear()
            self.ax_phase_space.set_title('Phase Space Trajectory')
            self.ax_phase_space.set_xlabel('Dimension 1')
            self.ax_phase_space.set_ylabel('Dimension 2')
            
            # Plot trajectory with color gradient
            points = len(trajectory)
            for i in range(1, points):
                alpha = i / points
                self.ax_phase_space.plot(trajectory[i-1:i+1, 0], 
                                       trajectory[i-1:i+1, 1],
                                       'b-', alpha=alpha)
            
            # Mark current position
            if points > 0:
                self.ax_phase_space.plot(trajectory[-1, 0], trajectory[-1, 1], 
                                       'ro', markersize=10)
        
        # Update chaos mode distribution
        self.update_chaos_modes()
        
        # Update soliton visualization
        self.update_soliton_viz()
        
        # Update attractor visualization
        self.update_attractor_viz()
        
        # Update status text
        self.update_status_text()
        
        return []
    
    def update_chaos_modes(self):
        """Update chaos mode distribution chart"""
        self.ax_chaos_modes.clear()
        self.ax_chaos_modes.set_title('Active Chaos Modes')
        
        # Get CCL status
        ccl_status = self.tori.ccl.get_status()
        active_tasks = ccl_status.get('active_tasks', 0)
        
        # Mock data for visualization (would be real in production)
        modes = ['Dark\nSoliton', 'Attractor\nHop', 'Phase\nExplosion']
        values = [
            min(active_tasks, 1) * np.random.random(),
            min(active_tasks, 1) * np.random.random(),
            min(active_tasks, 1) * np.random.random()
        ]
        
        colors = ['darkblue', 'darkgreen', 'darkred']
        bars = self.ax_chaos_modes.bar(modes, values, color=colors, alpha=0.7)
        self.ax_chaos_modes.set_ylim(0, 1)
        self.ax_chaos_modes.set_ylabel('Activity Level')
        
    def update_soliton_viz(self):
        """Update soliton visualization"""
        self.ax_soliton.clear()
        self.ax_soliton.set_title('Dark Soliton Profile')
        self.ax_soliton.set_xlabel('Position')
        self.ax_soliton.set_ylabel('Amplitude')
        
        # Generate soliton profile
        x = np.linspace(0, 100, 1000)
        
        # Dark soliton formula
        position = 50 + 10 * np.sin(len(self.timestamps) * 0.1)
        width = 10
        depth = 0.8
        
        amplitude = 1.0 - depth / np.cosh((x - position) / width) ** 2
        phase = np.pi * np.tanh((x - position) / width)
        
        # Plot amplitude
        self.ax_soliton.plot(x, amplitude, 'b-', linewidth=2, label='Amplitude')
        self.ax_soliton.fill_between(x, 0, amplitude, alpha=0.3)
        
        # Plot phase (scaled)
        self.ax_soliton.plot(x, 0.5 + phase / (2 * np.pi), 'r--', linewidth=1, label='Phase')
        
        self.ax_soliton.set_ylim(0, 1.5)
        self.ax_soliton.legend()
        
    def update_attractor_viz(self):
        """Update attractor basin visualization"""
        self.ax_attractor.clear()
        self.ax_attractor.set_title('Attractor Landscape')
        
        # Create attractor landscape
        x = np.linspace(-3, 3, 50)
        y = np.linspace(-3, 3, 50)
        X, Y = np.meshgrid(x, y)
        
        # Multiple attractors
        Z = np.zeros_like(X)
        attractors = [
            (0, 0, 1.0),
            (1.5, 1.5, 0.8),
            (-1.5, 1.5, 0.8),
            (0, -2, 0.6)
        ]
        
        for ax, ay, strength in attractors:
            Z -= strength * np.exp(-((X - ax)**2 + (Y - ay)**2) / 0.5)
        
        # Add current position from phase space
        if self.phase_space_trajectory:
            current = self.phase_space_trajectory[-1]
            Z -= 0.3 * np.exp(-((X - current[0])**2 + (Y - current[1])**2) / 0.3)
        
        # Plot landscape
        contour = self.ax_attractor.contourf(X, Y, Z, levels=20, cmap='viridis')
        self.ax_attractor.contour(X, Y, Z, levels=10, colors='black', alpha=0.3, linewidths=0.5)
        
        # Mark attractors
        for ax, ay, _ in attractors:
            self.ax_attractor.plot(ax, ay, 'wo', markersize=8, markeredgecolor='black')
        
        # Mark current position
        if self.phase_space_trajectory:
            self.ax_attractor.plot(current[0], current[1], 'ro', markersize=10)
            
    def update_status_text(self):
        """Update status text display"""
        status = self.tori.get_status()
        
        status_lines = [
            f"TORI Chaos-Enhanced System Status",
            f"{'='*35}",
            f"Mode: {status['adapter_mode']}",
            f"Safety: {status['safety']['current_safety_level']}",
            f"Chaos Events: {self.tori.stats['chaos_events']}",
            f"Queries: {self.tori.stats['queries_processed']}",
            f"",
            f"Energy Status:",
            f"  Available: {status['eigensentry']['energy_stats']['total_available']}",
            f"  Efficiency: {status['ccl']['efficiency_ratio']:.2f}x",
            f"",
            f"Active Components:",
            f"  EigenSentry: {'✓' if status['eigensentry']['stability_state'] else '✗'}",
            f"  CCL Tasks: {status['ccl']['active_tasks']}",
            f"  Checkpoints: {status['safety']['checkpoints_available']}"
        ]
        
        self.status_text.set_text('\n'.join(status_lines))
        
    def start(self):
        """Start the dashboard animation"""
        self.animation = FuncAnimation(
            self.fig, self.update_plots, interval=1000,  # Update every second
            blit=False, cache_frame_data=False
        )
        plt.show()
        
    def stop(self):
        """Stop the dashboard"""
        if self.animation:
            self.animation.event_source.stop()

# ========== Dashboard Runner ==========

async def run_dashboard():
    """Run the chaos dynamics dashboard"""
    print("🎨 Starting TORI Chaos Dynamics Dashboard...")
    
    # Initialize TORI system
    config = TORIProductionConfig(
        enable_chaos=True,
        default_adapter_mode=AdapterMode.HYBRID,
        enable_safety_monitoring=True
    )
    
    tori = TORIProductionSystem(config)
    await tori.start()
    
    # Create and start dashboard
    dashboard = ChaosDynamicsDashboard(tori)
    
    # Run some background queries to generate activity
    async def background_queries():
        queries = [
            "Explore patterns in complex systems",
            "Search for novel solutions",
            "Remember key insights",
            "Brainstorm creative approaches"
        ]
        
        while True:
            query = np.random.choice(queries)
            try:
                await tori.process_query(query, context={'enable_chaos': True})
            except:
                pass
            await asyncio.sleep(5)  # Query every 5 seconds
    
    # Start background task
    bg_task = asyncio.create_task(background_queries())
    
    try:
        # Run dashboard (blocks until window closed)
        dashboard.start()
    finally:
        # Cleanup
        bg_task.cancel()
        await tori.stop()

if __name__ == "__main__":
    # Run the dashboard
    asyncio.run(run_dashboard())
