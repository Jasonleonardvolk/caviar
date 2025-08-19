#!/usr/bin/env python3
"""
TORI Master Orchestrator
Integrates all chaos-enhanced components and work-streams
Provides unified launch and monitoring
"""

import asyncio
import sys
import os
import logging
import json
import signal
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess
import threading

# Add all necessary paths
sys.path.extend([
    str(Path(__file__).parent),
    str(Path(__file__).parent / "python" / "core"),
    str(Path(__file__).parent / "alan_backend"),
    str(Path(__file__).parent / "tools"),
    str(Path(__file__).parent / "services")
])

# Import all components
from python.core.tori_production import TORIProductionSystem, TORIProductionConfig
from python.core.metacognitive_adapters import AdapterMode
from alan_backend.eigensentry_guard import get_guard
from alan_backend.chaos_channel_controller import get_controller
from services.metrics_ws import MetricsWebSocketServer
from tools.simulate_darknet import DarkSolitonSimulator, load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tori_master.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TORIMaster:
    """
    Master orchestrator for the complete TORI system
    Manages all components and services
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "conf/master_config.yaml"
        self.running = False
        self.components = {}
        self.services = {}
        self.tasks = []
        
        # Load master configuration
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load master configuration"""
        default_config = {
            "enable_chaos": True,
            "enable_websocket": True,
            "enable_dark_solitons": True,
            "enable_ui_server": False,  # Set true if UI server needed
            "websocket_port": 8765,
            "ui_port": 3000,
            "lattice_config": "conf/lattice_config.yaml",
            "safety_monitoring": True,
            "feature_flags": {
                "CHAOS_EXPERIMENT": 1
            }
        }
        
        # TODO: Load from YAML if exists
        return default_config
        
    async def start(self):
        """Start all TORI components and services"""
        logger.info("🚀 Starting TORI Master System...")
        self.running = True
        
        try:
            # 1. Initialize core TORI production system
            logger.info("Initializing TORI production system...")
            tori_config = TORIProductionConfig(
                enable_chaos=self.config["enable_chaos"],
                enable_dark_solitons=self.config["enable_dark_solitons"],
                enable_safety_monitoring=self.config["safety_monitoring"]
            )
            self.components['tori'] = TORIProductionSystem(tori_config)
            await self.components['tori'].start()
            
            # 2. Initialize dark soliton simulator
            logger.info("Initializing dark soliton simulator...")
            lattice_config = load_config(self.config["lattice_config"])
            self.components['dark_soliton'] = DarkSolitonSimulator(lattice_config)
            
            # 3. Initialize chaos channel controller
            logger.info("Initializing chaos channel controller...")
            self.components['chaos_controller'] = get_controller()
            
            # 4. Initialize EigenSentry guard
            logger.info("Initializing EigenSentry guard...")
            self.components['eigen_guard'] = get_guard()
            
            # Wire BdG spectral stability
            if 'chaos_controller' in self.components and 'eigen_guard' in self.components:
                # Connect EigenSentry to CCL for adaptive timestep
                # Note: The chaos_controller from alan_backend doesn't have chaos_processor
                # but we can still log the connection attempt
                logger.info("✅ Wiring BdG spectral stability components")
                
                # If TORI has CCL component, wire it
                if hasattr(self.components['tori'], 'ccl'):
                    ccl = self.components['tori'].ccl
                    if hasattr(ccl, 'set_eigen_sentry'):
                        ccl.set_eigen_sentry(self.components['eigen_guard'])
                        logger.info("✅ Connected BdG spectral stability to chaos control")
            
            # 5. Start WebSocket metrics server
            if self.config["enable_websocket"]:
                logger.info("Starting WebSocket metrics server...")
                self.services['websocket'] = MetricsWebSocketServer()
                ws_task = asyncio.create_task(self.services['websocket'].start())
                self.tasks.append(ws_task)
                
            # 6. Start UI server if enabled
            if self.config.get("enable_ui_server"):
                logger.info("Starting UI server...")
                # This would start the React dev server or serve built files
                # For now, just log
                logger.info("UI server would start on port %d", self.config["ui_port"])
                
            # 7. Start system monitor
            logger.info("Starting system health monitor...")
            monitor_task = asyncio.create_task(self._monitor_system())
            self.tasks.append(monitor_task)
            
            # 8. Start integration handler
            logger.info("Starting integration handler...")
            integration_task = asyncio.create_task(self._handle_integrations())
            self.tasks.append(integration_task)
            
            logger.info("✅ All TORI components started successfully!")
            logger.info("System is ready for chaos-enhanced cognition")
            
            # Display status
            await self._display_status()
            
            # Wait for shutdown
            await self._wait_for_shutdown()
            
        except Exception as e:
            logger.error(f"Failed to start TORI: {e}")
            await self.stop()
            raise
            
    async def stop(self):
        """Stop all components gracefully"""
        logger.info("Stopping TORI Master System...")
        self.running = False
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
            
        # Wait for tasks to complete
        await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Stop TORI production system
        if 'tori' in self.components:
            await self.components['tori'].stop()
            
        logger.info("✅ TORI Master System stopped")
        
    async def _monitor_system(self):
        """Monitor system health"""
        while self.running:
            try:
                # Collect metrics from all components
                metrics = {
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'components': {}
                }
                
                # TORI status
                if 'tori' in self.components:
                    metrics['components']['tori'] = self.components['tori'].get_status()
                    
                # Chaos controller status
                if 'chaos_controller' in self.components:
                    controller = self.components['chaos_controller']
                    metrics['components']['chaos'] = {
                        'state': controller.state.value,
                        'energy': controller.current_energy,
                        'burst_history': len(controller.burst_history)
                    }
                    
                # EigenSentry status
                if 'eigen_guard' in self.components:
                    guard = self.components['eigen_guard']
                    metrics['components']['eigensentry'] = guard.metrics
                    
                # Log critical issues
                self._check_health(metrics)
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                await asyncio.sleep(5)
                
    def _check_health(self, metrics: Dict[str, Any]):
        """Check system health and log issues"""
        # Check TORI safety
        tori_metrics = metrics['components'].get('tori', {})
        safety = tori_metrics.get('safety', {})
        
        if safety.get('current_safety_level') == 'emergency':
            logger.error("⚠️ EMERGENCY: System safety critical!")
            
        # Check eigenvalues
        eigen_metrics = metrics['components'].get('eigensentry', {})
        if eigen_metrics.get('lyapunov_exponent', 0) > 0.5:
            logger.warning("⚠️ High Lyapunov exponent detected: %.3f", 
                         eigen_metrics['lyapunov_exponent'])
                         
        # Monitor Lyapunov exponents for system health (BdG)
        lambda_max = eigen_metrics.get('lambda_max', 0.0)
        if lambda_max > 0.1:
            logger.warning(f"⚠️ Positive Lyapunov exponent detected (BdG): {lambda_max:.3f}")
                         
    async def _handle_integrations(self):
        """Handle cross-component integrations"""
        while self.running:
            try:
                # Example: Connect dark soliton data to chaos controller
                if 'dark_soliton' in self.components and 'chaos_controller' in self.components:
                    # Get soliton field
                    soliton_field = self.components['dark_soliton'].get_field()
                    
                    # If chaos burst active, modulate solitons
                    controller = self.components['chaos_controller']
                    if controller.state.value == 'active':
                        # This would do actual integration
                        pass
                        
                await asyncio.sleep(0.1)  # 100ms integration cycle
                
            except Exception as e:
                logger.error(f"Integration error: {e}")
                await asyncio.sleep(1)
                
    async def _display_status(self):
        """Display system status"""
        print("\n" + "="*60)
        print("🌀 TORI CHAOS-ENHANCED SYSTEM STATUS")
        print("="*60)
        print(f"✅ Core System: Running")
        print(f"✅ Chaos Mode: {self.config['enable_chaos']}")
        print(f"✅ Dark Solitons: {self.config['enable_dark_solitons']}")
        print(f"✅ WebSocket Metrics: ws://localhost:{self.config['websocket_port']}/ws/eigensentry")
        print(f"✅ Safety Monitoring: {self.config['safety_monitoring']}")
        print("\nAvailable Commands:")
        print("  - Press Ctrl+C to shutdown")
        print("  - Check tori_master.log for details")
        print("  - Connect to WebSocket for live metrics")
        print("="*60 + "\n")
        
    async def _wait_for_shutdown(self):
        """Wait for shutdown signal"""
        shutdown_event = asyncio.Event()
        
        def signal_handler(sig, frame):
            logger.info("Shutdown signal received")
            shutdown_event.set()
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        await shutdown_event.wait()
        
    async def process_query(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Process a query through the integrated system"""
        if 'tori' not in self.components:
            raise RuntimeError("TORI not initialized")
            
        # Add integration context
        if context is None:
            context = {}
            
        # Enable features based on query
        if any(term in query.lower() for term in ['simulate', 'soliton', 'wave']):
            context['use_dark_solitons'] = True
            
        if any(term in query.lower() for term in ['explore', 'creative', 'brainstorm']):
            context['request_chaos_burst'] = True
            
        # Process through TORI
        result = await self.components['tori'].process_query(query, context)
        
        # If chaos burst was requested, trigger it
        if context.get('request_chaos_burst') and 'chaos_controller' in self.components:
            controller = self.components['chaos_controller']
            burst_id = controller.trigger(
                level=0.5,
                duration=100,
                purpose=f"query_exploration_{query[:20]}"
            )
            result['chaos_burst_id'] = burst_id
            
        return result

# Convenience functions
async def launch_tori(config_path: Optional[str] = None):
    """Launch TORI master system"""
    master = TORIMaster(config_path)
    await master.start()
    
def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="TORI Master System")
    parser.add_argument('--config', type=str, help='Path to master config file')
    parser.add_argument('--no-chaos', action='store_true', help='Disable chaos features')
    parser.add_argument('--no-websocket', action='store_true', help='Disable WebSocket server')
    
    args = parser.parse_args()
    
    # Create master with arguments
    master = TORIMaster(args.config)
    
    # Override config with command line args
    if args.no_chaos:
        master.config['enable_chaos'] = False
    if args.no_websocket:
        master.config['enable_websocket'] = False
        
    # Run
    try:
        asyncio.run(master.start())
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
