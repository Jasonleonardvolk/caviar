"""
ELFIN Stability Visualization Dashboard

This module provides a web-based dashboard for real-time monitoring of
stability properties, Lyapunov function values, and phase synchronization.
It uses Flask for the backend and D3.js for the frontend visualizations.
"""

import os
import json
import time
import threading
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

import numpy as np
from flask import Flask, render_template, jsonify, request, Response

try:
    from alan_backend.elfin.stability.lyapunov import LyapunovFunction
    from alan_backend.elfin.stability.jit_guard import StabilityGuard
except ImportError:
    # Minimal implementation for standalone testing
    class LyapunovFunction:
        def __init__(self, name):
            self.name = name
        
        def evaluate(self, x):
            return float(np.sum(np.array(x) ** 2))
    
    class StabilityGuard:
        def __init__(self, lyap, threshold=0):
            self.lyap = lyap
            self.threshold = threshold
            self.violations = 0


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DashboardServer:
    """Server for stability visualization dashboard."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5000,
        template_folder: Optional[str] = None,
        static_folder: Optional[str] = None
    ):
        """
        Initialize dashboard server.
        
        Args:
            host: Server host
            port: Server port
            template_folder: Folder for HTML templates
            static_folder: Folder for static files
        """
        self.host = host
        self.port = port
        
        # Setup template and static folders
        if template_folder is None:
            template_folder = os.path.join(
                os.path.dirname(__file__), "templates"
            )
        
        if static_folder is None:
            static_folder = os.path.join(
                os.path.dirname(__file__), "static"
            )
        
        # Ensure directories exist
        os.makedirs(template_folder, exist_ok=True)
        os.makedirs(static_folder, exist_ok=True)
        
        # Create Flask app
        self.app = Flask(
            __name__,
            template_folder=template_folder,
            static_folder=static_folder
        )
        
        # Stability data
        self.lyapunov_functions = {}
        self.stability_guards = {}
        self.phase_states = {}
        
        # Simulation data
        self.system_states = {}
        self.history = {
            "lyapunov_values": {},
            "system_states": {},
            "phase_states": {},
            "stability_violations": {}
        }
        
        # Max history length
        self.max_history = 1000
        
        # Register routes
        self.register_routes()
        
        # Server thread
        self.server_thread = None
        self.running = False
    
    def register_routes(self):
        """Register Flask routes."""
        # Main dashboard route
        @self.app.route("/")
        def index():
            return render_template("dashboard.html")
        
        # API routes
        @self.app.route("/api/data")
        def get_data():
            return jsonify({
                "lyapunov_functions": {
                    name: {"name": name}
                    for name in self.lyapunov_functions
                },
                "stability_guards": {
                    name: {
                        "name": name,
                        "violations": guard.violations,
                        "threshold": guard.threshold
                    }
                    for name, guard in self.stability_guards.items()
                },
                "phase_states": {
                    name: {"name": name}
                    for name in self.phase_states
                },
                "system_states": self.system_states
            })
        
        @self.app.route("/api/history")
        def get_history():
            return jsonify(self.history)
        
        @self.app.route("/api/lyapunov")
        def get_lyapunov():
            name = request.args.get("name")
            if name not in self.lyapunov_functions:
                return jsonify({"error": f"Unknown Lyapunov function: {name}"}), 404
            
            return jsonify({
                "name": name,
                "history": self.history["lyapunov_values"].get(name, [])
            })
        
        @self.app.route("/api/stability")
        def get_stability():
            name = request.args.get("name")
            if name not in self.stability_guards:
                return jsonify({"error": f"Unknown stability guard: {name}"}), 404
            
            guard = self.stability_guards[name]
            return jsonify({
                "name": name,
                "violations": guard.violations,
                "threshold": guard.threshold,
                "history": self.history["stability_violations"].get(name, [])
            })
        
        @self.app.route("/api/phase")
        def get_phase():
            name = request.args.get("name")
            if name not in self.phase_states:
                return jsonify({"error": f"Unknown phase state: {name}"}), 404
            
            return jsonify({
                "name": name,
                "history": self.history["phase_states"].get(name, [])
            })
        
        @self.app.route("/api/system")
        def get_system():
            name = request.args.get("name", "default")
            if name not in self.system_states:
                return jsonify({"error": f"Unknown system state: {name}"}), 404
            
            return jsonify({
                "name": name,
                "state": self.system_states.get(name),
                "history": self.history["system_states"].get(name, [])
            })
        
        # SSE stream for real-time updates
        @self.app.route("/api/stream")
        def stream():
            def event_stream():
                last_update = 0
                while True:
                    # Check if there's new data (every 100ms)
                    time.sleep(0.1)
                    
                    # Send update every second
                    if time.time() - last_update > 1.0:
                        last_update = time.time()
                        data = {
                            "lyapunov_values": {
                                name: self._get_lyapunov_value(name)
                                for name in self.lyapunov_functions
                            },
                            "stability_violations": {
                                name: guard.violations
                                for name, guard in self.stability_guards.items()
                            },
                            "system_states": self.system_states,
                            "timestamp": time.time()
                        }
                        yield f"data: {json.dumps(data)}\n\n"
            
            return Response(
                event_stream(),
                mimetype="text/event-stream"
            )
    
    def _get_lyapunov_value(self, name: str) -> float:
        """
        Get current Lyapunov function value.
        
        Args:
            name: Lyapunov function name
            
        Returns:
            Current Lyapunov value
        """
        if name not in self.lyapunov_functions:
            return 0.0
        
        lyap = self.lyapunov_functions[name]
        
        # Find system state for this Lyapunov function
        system_name = "default"
        for s_name, state in self.system_states.items():
            if state is not None:
                system_name = s_name
                break
        
        state = self.system_states.get(system_name)
        if state is None:
            return 0.0
        
        # Evaluate Lyapunov function
        return lyap.evaluate(state)
    
    def register_lyapunov_function(self, lyap: LyapunovFunction):
        """
        Register a Lyapunov function for monitoring.
        
        Args:
            lyap: Lyapunov function
        """
        name = getattr(lyap, "name", str(lyap))
        self.lyapunov_functions[name] = lyap
        
        # Initialize history
        if name not in self.history["lyapunov_values"]:
            self.history["lyapunov_values"][name] = []
    
    def register_stability_guard(self, guard: StabilityGuard, name: Optional[str] = None):
        """
        Register a stability guard for monitoring.
        
        Args:
            guard: Stability guard
            name: Optional name (defaults to lyap.name)
        """
        if name is None:
            name = getattr(guard.lyap, "name", str(guard.lyap))
        
        self.stability_guards[name] = guard
        
        # Initialize history
        if name not in self.history["stability_violations"]:
            self.history["stability_violations"][name] = []
    
    def register_phase_state(self, phase_state, name: Optional[str] = None):
        """
        Register a phase state for monitoring.
        
        Args:
            phase_state: Phase state
            name: Optional name
        """
        if name is None:
            name = str(phase_state)
        
        self.phase_states[name] = phase_state
        
        # Initialize history
        if name not in self.history["phase_states"]:
            self.history["phase_states"][name] = []
    
    def update_system_state(self, state, name: str = "default"):
        """
        Update system state.
        
        Args:
            state: System state
            name: System name
        """
        # Convert to list for JSON serialization
        if hasattr(state, "tolist"):
            state = state.tolist()
        
        # Update state
        self.system_states[name] = state
        
        # Update history
        if name not in self.history["system_states"]:
            self.history["system_states"][name] = []
        
        self.history["system_states"][name].append({
            "time": time.time(),
            "state": state
        })
        
        # Trim history
        if len(self.history["system_states"][name]) > self.max_history:
            self.history["system_states"][name] = self.history["system_states"][name][-self.max_history:]
        
        # Update Lyapunov values
        for lyap_name, lyap in self.lyapunov_functions.items():
            value = lyap.evaluate(state)
            
            if lyap_name not in self.history["lyapunov_values"]:
                self.history["lyapunov_values"][lyap_name] = []
            
            self.history["lyapunov_values"][lyap_name].append({
                "time": time.time(),
                "value": value
            })
            
            # Trim history
            if len(self.history["lyapunov_values"][lyap_name]) > self.max_history:
                self.history["lyapunov_values"][lyap_name] = self.history["lyapunov_values"][lyap_name][-self.max_history:]
    
    def record_stability_violation(self, guard_name: str, x_prev, x):
        """
        Record a stability violation.
        
        Args:
            guard_name: Stability guard name
            x_prev: Previous state
            x: Current state
        """
        if guard_name not in self.history["stability_violations"]:
            self.history["stability_violations"][guard_name] = []
        
        # Convert to list for JSON serialization
        if hasattr(x_prev, "tolist"):
            x_prev = x_prev.tolist()
        if hasattr(x, "tolist"):
            x = x.tolist()
        
        self.history["stability_violations"][guard_name].append({
            "time": time.time(),
            "x_prev": x_prev,
            "x": x
        })
        
        # Trim history
        if len(self.history["stability_violations"][guard_name]) > self.max_history:
            self.history["stability_violations"][guard_name] = self.history["stability_violations"][guard_name][-self.max_history:]
    
    def start(self, debug: bool = False):
        """
        Start dashboard server.
        
        Args:
            debug: Whether to run in debug mode
        """
        if self.running:
            logger.warning("Server already running")
            return
        
        self.running = True
        
        if self.server_thread is None:
            self.server_thread = threading.Thread(
                target=self._run_server,
                args=(debug,)
            )
            self.server_thread.daemon = True
            self.server_thread.start()
        
        logger.info(f"Dashboard server started at http://{self.host}:{self.port}")
    
    def _run_server(self, debug: bool):
        """Run Flask server."""
        self.app.run(
            host=self.host,
            port=self.port,
            debug=debug,
            use_reloader=False
        )
    
    def stop(self):
        """Stop dashboard server."""
        self.running = False
        self.server_thread = None
        logger.info("Dashboard server stopped")


def create_dashboard_files():
    """Create dashboard HTML and JS files."""
    # Get directory of this script
    dir_path = os.path.dirname(os.path.abspath(__file__))
    
    # Create templates directory
    templates_dir = os.path.join(dir_path, "templates")
    os.makedirs(templates_dir, exist_ok=True)
    
    # Create static directory
    static_dir = os.path.join(dir_path, "static")
    os.makedirs(static_dir, exist_ok=True)
    
    # Create dashboard.html
    dashboard_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ELFIN Stability Dashboard</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css">
    <link rel="stylesheet" href="{{ url_for('static', filename='dashboard.css') }}">
    <script src="https://d3js.org/d3.v7.min.js"></script>
</head>
<body>
    <nav class="navbar navbar-dark bg-dark">
        <div class="container-fluid">
            <span class="navbar-brand">ELFIN Stability Dashboard</span>
            <div class="d-flex">
                <div class="status-indicator" id="connection-status"></div>
                <span class="text-light ms-2">Connection</span>
            </div>
        </div>
    </nav>
    
    <div class="container-fluid mt-3">
        <div class="row">
            <div class="col-md-6">
                <div class="card mb-3">
                    <div class="card-header">
                        Lyapunov Function Values
                    </div>
                    <div class="card-body">
                        <div id="lyapunov-chart" class="chart-container"></div>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card mb-3">
                    <div class="card-header">
                        System State
                    </div>
                    <div class="card-body">
                        <div id="state-chart" class="chart-container"></div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row">
            <div class="col-md-6">
                <div class="card mb-3">
                    <div class="card-header">
                        Stability Violations
                    </div>
                    <div class="card-body">
                        <div id="violations-chart" class="chart-container"></div>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card mb-3">
                    <div class="card-header">
                        Phase Space
                    </div>
                    <div class="card-body">
                        <div id="phase-chart" class="chart-container"></div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row">
            <div class="col-md-12">
                <div class="card mb-3">
                    <div class="card-header">
                        Stability Metrics
                    </div>
                    <div class="card-body">
                        <div id="metrics" class="row"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="{{ url_for('static', filename='dashboard.js') }}"></script>
</body>
</html>
"""
    
    with open(os.path.join(templates_dir, "dashboard.html"), "w") as f:
        f.write(dashboard_html)
    
    # Create dashboard.css
    dashboard_css = """.chart-container {
    height: 300px;
    width: 100%;
}

.status-indicator {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background-color: red;
    margin-top: 5px;
}

.status-indicator.connected {
    background-color: green;
}

.metric-card {
    padding: 15px;
    border-radius: 5px;
    margin-bottom: 10px;
}

.metric-value {
    font-size: 2em;
    font-weight: bold;
}

.metric-title {
    font-size: 0.9em;
    color: #666;
}

.violation-alert {
    background-color: #f8d7da;
    color: #721c24;
}

.stable-alert {
    background-color: #d4edda;
    color: #155724;
}
"""
    
    with open(os.path.join(static_dir, "dashboard.css"), "w") as f:
        f.write(dashboard_css)
    
    # Create dashboard.js
    dashboard_js = """// Dashboard State
let state = {
    lyapunovValues: {},
    systemStates: {},
    stabilityViolations: {},
    phaseStates: {},
    connected: false
};

// Colors for different lines
const colors = d3.schemeCategory10;

// Setup SSE for real-time updates
const eventSource = new EventSource("/api/stream");
eventSource.onopen = function() {
    state.connected = true;
    document.getElementById("connection-status").classList.add("connected");
};

eventSource.onerror = function() {
    state.connected = false;
    document.getElementById("connection-status").classList.remove("connected");
};

eventSource.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    // Update state
    Object.entries(data.lyapunov_values).forEach(([name, value]) => {
        if (!state.lyapunovValues[name]) {
            state.lyapunovValues[name] = [];
        }
        
        state.lyapunovValues[name].push({
            time: data.timestamp,
            value: value
        });
        
        // Keep only the last 100 points
        if (state.lyapunovValues[name].length > 100) {
            state.lyapunovValues[name].shift();
        }
    });
    
    Object.entries(data.stability_violations).forEach(([name, violations]) => {
        if (!state.stabilityViolations[name]) {
            state.stabilityViolations[name] = [];
        }
        
        // Record violation count
        state.stabilityViolations[name].push({
            time: data.timestamp,
            count: violations
        });
        
        // Keep only the last 100 points
        if (state.stabilityViolations[name].length > 100) {
            state.stabilityViolations[name].shift();
        }
    });
    
    Object.entries(data.system_states).forEach(([name, stateValue]) => {
        if (!state.systemStates[name]) {
            state.systemStates[name] = [];
        }
        
        state.systemStates[name].push({
            time: data.timestamp,
            state: stateValue
        });
        
        // Keep only the last 100 points
        if (state.systemStates[name].length > 100) {
            state.systemStates[name].shift();
        }
    });
    
    // Update charts
    updateLyapunovChart();
    updateViolationsChart();
    updateStateChart();
    updatePhaseChart();
    updateMetrics();
};

// Fetch initial data
fetch("/api/data")
    .then(response => response.json())
    .then(data => {
        // Setup charts
        setupLyapunovChart(Object.keys(data.lyapunov_functions));
        setupViolationsChart(Object.keys(data.stability_guards));
        setupStateChart(Object.keys(data.system_states));
        setupPhaseChart();
        setupMetrics(data);
    });

// Setup Lyapunov Chart
let lyapunovChart = {
    svg: null,
    width: 0,
    height: 0,
    x: null,
    y: null,
    line: null,
    functions: []
};

function setupLyapunovChart(functions) {
    lyapunovChart.functions = functions;
    
    const container = d3.select("#lyapunov-chart");
    const width = container.node().clientWidth;
    const height = container.node().clientHeight;
    
    lyapunovChart.width = width;
    lyapunovChart.height = height;
    
    // Create SVG
    lyapunovChart.svg = container.append("svg")
        .attr("width", width)
        .attr("height", height);
    
    // Create scales
    lyapunovChart.x = d3.scaleLinear()
        .domain([0, 100])
        .range([50, width - 20]);
    
    lyapunovChart.y = d3.scaleLinear()
        .domain([0, 10])
        .range([height - 30, 20]);
    
    // Create axes
    lyapunovChart.svg.append("g")
        .attr("class", "x-axis")
        .attr("transform", `translate(0, ${height - 30})`)
        .call(d3.axisBottom(lyapunovChart.x));
    
    lyapunovChart.svg.append("g")
        .attr("class", "y-axis")
        .attr("transform", "translate(50, 0)")
        .call(d3.axisLeft(lyapunovChart.y));
    
    // Create line generator
    lyapunovChart.line = d3.line()
        .x(d => lyapunovChart.x(d.time))
        .y(d => lyapunovChart.y(d.value));
    
    // Add title
    lyapunovChart.svg.append("text")
        .attr("x", width / 2)
        .attr("y", 10)
        .attr("text-anchor", "middle")
        .style("font-size", "12px")
        .text("Lyapunov Function Values");
    
    // Add legend
    const legend = lyapunovChart.svg.append("g")
        .attr("class", "legend")
        .attr("transform", `translate(${width - 100}, 20)`);
    
    functions.forEach((func, i) => {
        legend.append("rect")
            .attr("x", 0)
            .attr("y", i * 20)
            .attr("width", 10)
            .attr("height", 10)
            .style("fill", colors[i % colors.length]);
        
        legend.append("text")
            .attr("x", 15)
            .attr("y", i * 20 + 9)
            .style("font-size", "10px")
            .text(func);
    });
}

function updateLyapunovChart() {
    if (!lyapunovChart.svg) return;
    
    // Update scales
    let maxValue = 0;
    Object.values(state.lyapunovValues).forEach(values => {
        if (values.length > 0) {
            const localMax = d3.max(values, d => d.value);
            maxValue = Math.max(maxValue, localMax);
        }
    });
    
    lyapunovChart.y.domain([0, maxValue * 1.1 || 10]);
    
    // Update axes
    lyapunovChart.svg.select(".y-axis")
        .call(d3.axisLeft(lyapunovChart.y));
    
    // Update lines
    lyapunovChart.functions.forEach((func, i) => {
        const values = state.lyapunovValues[func] || [];
        
        // Check if line exists
        let line = lyapunovChart.svg.select(`.line-${func}`);
        
        if (line.empty()) {
            // Create new line
            line = lyapunovChart.svg.append("path")
                .attr("class", `line-${func}`)
                .style("fill", "none")
                .style("stroke", colors[i % colors.length])
                .style("stroke-width", 2);
        }
        
        // Update line
        if (values.length > 0) {
            // Normalize time
            const normalizedValues = values.map((d, i) => ({
                time: i,
                value: d.value
            }));
            
            line.datum(normalizedValues)
                .attr("d", lyapunovChart.line);
        }
    });
}

// Setup Violations Chart
let violationsChart = {
    svg: null,
    width: 0,
    height: 0,
    x: null,
    y: null,
    line: null,
    guards: []
};

function setupViolationsChart(guards) {
    violationsChart.guards = guards;
    
    const container = d3.select("#violations-chart");
    const width = container.node().clientWidth;
    const height = container.node().clientHeight;
    
    violationsChart.width = width;
    violationsChart.height = height;
    
    // Create SVG
    violationsChart.svg = container.append("svg")
        .attr("width", width)
        .attr("height", height);
    
    // Create scales
    violationsChart.x = d3.scaleLinear()
        .domain([0, 100])
        .range([50, width - 20]);
    
    violationsChart.y = d3.scaleLinear()
        .domain([0, 10])
        .range([height - 30, 20]);
    
    // Create axes
    violationsChart.svg.append("g")
        .attr("class", "x-axis")
        .attr("transform", `translate(0, ${height - 30})`)
        .call(d3.axisBottom(violationsChart.x));
    
    violationsChart.svg.append("g")
        .attr("class", "y-axis")
        .attr("transform", "translate(50, 0)")
        .call(d3.axisLeft(violationsChart.y));
    
    // Create line generator
    violationsChart.line = d3.line()
        .x(d => violationsChart.x(d.time))
        .y(d => violationsChart.y(d.count));
    
    // Add title
    violationsChart.svg.append("text")
        .attr("x", width / 2)
        .attr("y", 10)
        .attr("text-anchor", "middle")
        .style("font-size", "12px")
        .text("Stability Violations");
    
    // Add legend
    const legend = violationsChart.svg.append("g")
        .attr("class", "legend")
        .attr("transform", `translate(${width - 100}, 20)`);
    
    guards.forEach((guard, i) => {
        legend.append("rect")
            .attr("x", 0)
            .attr("y", i * 20)
            .attr("width", 10)
            .attr("height", 10)
            .style("fill", colors[i % colors.length]);
        
        legend.append("text")
            .attr("x", 15)
            .attr("y", i * 20 + 9)
            .style("font-size", "10px")
            .text(guard);
    });
}

function updateViolationsChart() {
    if (!violationsChart.svg) return;
    
    // Update scales
    let maxValue = 0;
    Object.values(state.stabilityViolations).forEach(values => {
        if (values.length > 0) {
            const localMax = d3.max(values, d => d.count);
            maxValue = Math.max(maxValue, localMax);
        }
    });
    
    violationsChart.y.domain([0, maxValue * 1.1 || 10]);
    
    // Update axes
    violationsChart.svg.select(".y-axis")
        .call(d3.axisLeft(violationsChart.y));
    
    // Update lines
    violationsChart.guards.forEach((guard, i) => {
        const values = state.stabilityViolations[guard] || [];
        
        // Check if line exists
        let line = violationsChart.svg.select(`.line-${guard}`);
        
        if (line.empty()) {
            // Create new line
            line = violationsChart.svg.append("path")
                .attr("class", `line-${guard}`)
                .style("fill", "none")
                .style("stroke", colors[i % colors.length])
                .style("stroke-width", 2);
        }
        
        // Update line
        if (values.length > 0) {
            // Normalize time
            const normalizedValues = values.map((d, i) => ({
                time: i,
                count: d.count
            }));
            
            line.datum(normalizedValues)
                .attr("d", violationsChart.line);
        }
    });
}

// Setup State Chart
let stateChart = {
    svg: null,
    width: 0,
    height: 0,
    x: null,
    y: null,
    line: null,
    systems: []
};

function setupStateChart(systems) {
    stateChart.systems = systems;
    
    const container = d3.select("#state-chart");
