#!/usr/bin/env python3
"""
TORI System Health Monitor
Real-time monitoring dashboard for all components
"""

import asyncio
import curses
from datetime import datetime, timezone
import json
import websockets
from typing import Dict, Any, Optional
import numpy as np
from collections import deque
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

class TORIHealthMonitor:
    """
    Terminal-based health monitoring dashboard
    Shows real-time status of all TORI components
    """
    
    def __init__(self):
        self.running = False
        self.metrics = {
            'tori': {'status': 'Unknown'},
            'eigensentry': {'status': 'Unknown'},
            'chaos': {'status': 'Unknown'},
            'soliton': {'status': 'Unknown'},
            'websocket': {'status': 'Unknown'}
        }
        
        # History buffers
        self.eigenvalue_history = deque(maxlen=50)
        self.energy_history = deque(maxlen=50)
        self.safety_history = deque(maxlen=50)
        
        # WebSocket connection
        self.websocket = None
        
    async def start(self):
        """Start the monitoring dashboard"""
        self.running = True
        
        # Start WebSocket listener
        ws_task = asyncio.create_task(self.websocket_listener())
        
        # Start curses interface
        try:
            curses.wrapper(self.run_dashboard)
        finally:
            self.running = False
            ws_task.cancel()
            if self.websocket:
                await self.websocket.close()
                
    def run_dashboard(self, stdscr):
        """Run the curses dashboard"""
        # Setup
        curses.curs_set(0)  # Hide cursor
        stdscr.nodelay(1)   # Non-blocking input
        stdscr.timeout(100) # Refresh rate
        
        # Colors
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)
        
        while self.running:
            stdscr.clear()
            height, width = stdscr.getmaxyx()
            
            # Header
            self.draw_header(stdscr, width)
            
            # Component Status
            self.draw_component_status(stdscr, 2, 0, width)
            
            # Metrics Graphs
            self.draw_metrics(stdscr, 10, 0, width, height - 15)
            
            # Footer
            self.draw_footer(stdscr, height - 2, width)
            
            stdscr.refresh()
            
            # Check for quit
            key = stdscr.getch()
            if key == ord('q') or key == ord('Q'):
                self.running = False
                
    def draw_header(self, stdscr, width):
        """Draw dashboard header"""
        title = "🌀 TORI CHAOS-ENHANCED SYSTEM HEALTH MONITOR"
        stdscr.attron(curses.color_pair(4))
        stdscr.addstr(0, (width - len(title)) // 2, title)
        stdscr.attroff(curses.color_pair(4))
        
    def draw_component_status(self, stdscr, y, x, width):
        """Draw component status panel"""
        stdscr.addstr(y, x, "COMPONENT STATUS:")
        
        components = [
            ("TORI Core", self.get_status_color('tori')),
            ("EigenSentry", self.get_status_color('eigensentry')),
            ("Chaos Controller", self.get_status_color('chaos')),
            ("Dark Solitons", self.get_status_color('soliton')),
            ("WebSocket", self.get_status_color('websocket'))
        ]
        
        col_width = width // len(components)
        for i, (name, color) in enumerate(components):
            status = self.metrics.get(name.lower().split()[0], {}).get('status', 'Unknown')
            
            stdscr.attron(curses.color_pair(color))
            stdscr.addstr(y + 2, x + i * col_width, f"[{status:^8}]")
            stdscr.attroff(curses.color_pair(color))
            stdscr.addstr(y + 3, x + i * col_width, name[:col_width-1])
            
    def get_status_color(self, component: str) -> int:
        """Get color for component status"""
        status = self.metrics.get(component, {}).get('status', 'Unknown')
        if status in ['Active', 'Running', 'Connected']:
            return 1  # Green
        elif status in ['Idle', 'Cooldown']:
            return 2  # Yellow
        elif status in ['Error', 'Critical']:
            return 3  # Red
        else:
            return 4  # Cyan
            
    def draw_metrics(self, stdscr, y, x, width, height):
        """Draw metrics graphs"""
        stdscr.addstr(y, x, "SYSTEM METRICS:")
        
        # Split into 3 columns
        col_width = width // 3
        graph_height = height - 2
        
        # Eigenvalue graph
        self.draw_graph(stdscr, y + 2, x, col_width - 2, graph_height,
                       "Eigenvalues", self.eigenvalue_history, 2.0)
                       
        # Energy graph  
        self.draw_graph(stdscr, y + 2, x + col_width, col_width - 2, graph_height,
                       "Energy", self.energy_history, 5.0)
                       
        # Safety graph
        self.draw_graph(stdscr, y + 2, x + 2 * col_width, col_width - 2, graph_height,
                       "Safety Score", self.safety_history, 1.0)
                       
    def draw_graph(self, stdscr, y, x, width, height, title, data, max_val):
        """Draw a simple ASCII graph"""
        # Title
        stdscr.addstr(y, x + (width - len(title)) // 2, title)
        
        # Border
        for i in range(height):
            stdscr.addstr(y + 1 + i, x, "│")
            stdscr.addstr(y + 1 + i, x + width - 1, "│")
        stdscr.addstr(y + height, x, "└" + "─" * (width - 2) + "┘")
        
        # Plot data
        if data:
            # Scale data
            scaled_data = [min(d / max_val, 1.0) * (height - 2) for d in data]
            
            # Draw points
            for i, val in enumerate(scaled_data[-width+2:]):
                if i < width - 2:
                    bar_height = int(val)
                    for j in range(bar_height):
                        stdscr.addstr(y + height - 1 - j, x + 1 + i, "█")
                        
    def draw_footer(self, stdscr, y, width):
        """Draw footer with controls"""
        footer = "Press 'q' to quit | Updated: " + datetime.now().strftime("%H:%M:%S")
        stdscr.addstr(y, (width - len(footer)) // 2, footer)
        
    async def websocket_listener(self):
        """Listen to WebSocket metrics"""
        uri = "ws://localhost:8765/ws/eigensentry"
        
        while self.running:
            try:
                async with websockets.connect(uri) as websocket:
                    self.websocket = websocket
                    self.metrics['websocket']['status'] = 'Connected'
                    
                    async for message in websocket:
                        data = json.loads(message)
                        self.process_websocket_data(data)
                        
            except Exception as e:
                self.metrics['websocket']['status'] = 'Error'
                await asyncio.sleep(5)  # Retry after 5 seconds
                
    def process_websocket_data(self, data: Dict[str, Any]):
        """Process WebSocket data"""
        msg_type = data.get('type')
        
        if msg_type == 'metrics_update':
            metrics = data.get('data', {})
            
            # Update eigenvalue history
            max_eigen = metrics.get('max_eigenvalue', 0)
            self.eigenvalue_history.append(max_eigen)
            
            # Update component status
            if max_eigen > 2.0:
                self.metrics['eigensentry']['status'] = 'Critical'
            elif max_eigen > 1.3:
                self.metrics['eigensentry']['status'] = 'Active'
            else:
                self.metrics['eigensentry']['status'] = 'Stable'
                
            # Update other metrics
            if metrics.get('damping_active'):
                self.metrics['chaos']['status'] = 'Damping'
            else:
                self.metrics['chaos']['status'] = 'Idle'

# Alternative simple monitor for non-curses environments
class SimpleHealthMonitor:
    """Simple text-based monitor for environments without curses"""
    
    def __init__(self):
        self.running = False
        
    async def start(self):
        """Start simple monitoring"""
        self.running = True
        
        print("TORI System Health Monitor (Simple Mode)")
        print("Press Ctrl+C to stop")
        print("-" * 60)
        
        uri = "ws://localhost:8765/ws/eigensentry"
        
        try:
            async with websockets.connect(uri) as websocket:
                async for message in websocket:
                    if not self.running:
                        break
                        
                    data = json.loads(message)
                    if data.get('type') == 'metrics_update':
                        metrics = data.get('data', {})
                        
                        # Clear line and print metrics
                        print(f"\r[{datetime.now().strftime('%H:%M:%S')}] "
                              f"Eigen: {metrics.get('max_eigenvalue', 0):.3f} | "
                              f"Lyapunov: {metrics.get('lyapunov_exponent', 0):.3f} | "
                              f"Damping: {'ON' if metrics.get('damping_active') else 'OFF'} | "
                              f"Curvature: {metrics.get('mean_curvature', 0):.3f}",
                              end='', flush=True)
                              
        except KeyboardInterrupt:
            print("\nMonitor stopped")
        except Exception as e:
            print(f"\nError: {e}")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="TORI Health Monitor")
    parser.add_argument('--simple', action='store_true', 
                       help='Use simple text mode instead of curses')
    
    args = parser.parse_args()
    
    if args.simple or sys.platform == 'win32':
        # Use simple monitor on Windows or when requested
        monitor = SimpleHealthMonitor()
    else:
        monitor = TORIHealthMonitor()
        
    try:
        asyncio.run(monitor.start())
    except KeyboardInterrupt:
        print("\nMonitor stopped")

if __name__ == "__main__":
    main()
