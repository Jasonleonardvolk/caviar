// Dashboard State
const state = {
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
        
        if (stateValue) {
            state.systemStates[name].push({
                time: data.timestamp,
                state: stateValue
            });
            
            // Keep only the last 100 points
            if (state.systemStates[name].length > 100) {
                state.systemStates[name].shift();
            }
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
const lyapunovChart = {
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
const violationsChart = {
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
const stateChart = {
    svg: null,
    width: 0,
    height: 0,
    x: null,
    y: null,
    lines: [],
    systems: []
};

function setupStateChart(systems) {
    stateChart.systems = systems;
    
    const container = d3.select("#state-chart");
    const width = container.node().clientWidth;
    const height = container.node().clientHeight;
    
    stateChart.width = width;
    stateChart.height = height;
    
    // Create SVG
    stateChart.svg = container.append("svg")
        .attr("width", width)
        .attr("height", height);
    
    // Create scales
    stateChart.x = d3.scaleLinear()
        .domain([0, 100])
        .range([50, width - 20]);
    
    stateChart.y = d3.scaleLinear()
        .domain([-5, 5])
        .range([height - 30, 20]);
    
    // Create axes
    stateChart.svg.append("g")
        .attr("class", "x-axis")
        .attr("transform", `translate(0, ${height - 30})`)
        .call(d3.axisBottom(stateChart.x));
    
    stateChart.svg.append("g")
        .attr("class", "y-axis")
        .attr("transform", "translate(50, 0)")
        .call(d3.axisLeft(stateChart.y));
    
    // Create line generator for each state variable
    stateChart.lines = [];
    
    // Add title
    stateChart.svg.append("text")
        .attr("x", width / 2)
        .attr("y", 10)
        .attr("text-anchor", "middle")
        .style("font-size", "12px")
        .text("System State");
}

function updateStateChart() {
    if (!stateChart.svg) return;
    
    // Get all state variables
    const stateVariables = new Set();
    let minValue = 0;
    let maxValue = 0;
    
    Object.entries(state.systemStates).forEach(([sysName, sysStates]) => {
        if (sysStates.length > 0 && sysStates[0].state) {
            const lastState = sysStates[sysStates.length - 1].state;
            if (Array.isArray(lastState)) {
                for (let i = 0; i < lastState.length; i++) {
                    stateVariables.add(`${sysName}_${i}`);
                }
                
                // Find min/max
                sysStates.forEach(data => {
                    if (Array.isArray(data.state)) {
                        data.state.forEach(val => {
                            minValue = Math.min(minValue, val);
                            maxValue = Math.max(maxValue, val);
                        });
                    }
                });
            }
        }
    });
    
    // Update scales with some padding
    stateChart.y.domain([minValue * 1.1, maxValue * 1.1]);
    
    // Update axes
    stateChart.svg.select(".y-axis")
        .call(d3.axisLeft(stateChart.y));
    
    // Create line generator
    const line = d3.line()
        .x(d => stateChart.x(d.time))
        .y(d => stateChart.y(d.value));
    
    // Update lines
    Array.from(stateVariables).forEach((varName, i) => {
        const [sysName, varIndex] = varName.split('_');
        const varIdx = parseInt(varIndex);
        
        // Extract values for this variable
        const values = [];
        const sysStates = state.systemStates[sysName] || [];
        
        sysStates.forEach((data, j) => {
            if (Array.isArray(data.state) && varIdx < data.state.length) {
                values.push({
                    time: j,
                    value: data.state[varIdx]
                });
            }
        });
        
        // Check if line exists
        let varLine = stateChart.svg.select(`.line-${varName}`);
        
        if (varLine.empty()) {
            // Create new line
            varLine = stateChart.svg.append("path")
                .attr("class", `line-${varName}`)
                .style("fill", "none")
                .style("stroke", colors[i % colors.length])
                .style("stroke-width", 2);
            
            // Add to legend if not exists
            const legend = stateChart.svg.select(".state-legend");
            if (legend.empty()) {
                const newLegend = stateChart.svg.append("g")
                    .attr("class", "state-legend")
                    .attr("transform", `translate(${stateChart.width - 100}, 20)`);
                
                newLegend.append("rect")
                    .attr("x", 0)
                    .attr("y", i * 20)
                    .attr("width", 10)
                    .attr("height", 10)
                    .style("fill", colors[i % colors.length]);
                
                newLegend.append("text")
                    .attr("x", 15)
                    .attr("y", i * 20 + 9)
                    .style("font-size", "10px")
                    .text(`${sysName}[${varIdx}]`);
            } else {
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
                    .text(`${sysName}[${varIdx}]`);
            }
        }
        
        // Update line
        if (values.length > 0) {
            varLine.datum(values)
                .attr("d", line);
        }
    });
}

// Setup Phase Chart
const phaseChart = {
    svg: null,
    width: 0,
    height: 0,
    x: null,
    y: null
};

function setupPhaseChart() {
    const container = d3.select("#phase-chart");
    const width = container.node().clientWidth;
    const height = container.node().clientHeight;
    
    phaseChart.width = width;
    phaseChart.height = height;
    
    // Create SVG
    phaseChart.svg = container.append("svg")
        .attr("width", width)
        .attr("height", height);
    
    // Create scales
    phaseChart.x = d3.scaleLinear()
        .domain([-5, 5])
        .range([50, width - 20]);
    
    phaseChart.y = d3.scaleLinear()
        .domain([-5, 5])
        .range([height - 30, 20]);
    
    // Create axes
    phaseChart.svg.append("g")
        .attr("class", "x-axis")
        .attr("transform", `translate(0, ${height - 30})`)
        .call(d3.axisBottom(phaseChart.x));
    
    phaseChart.svg.append("g")
        .attr("class", "y-axis")
        .attr("transform", "translate(50, 0)")
        .call(d3.axisLeft(phaseChart.y));
    
    // Add title
    phaseChart.svg.append("text")
        .attr("x", width / 2)
        .attr("y", 10)
        .attr("text-anchor", "middle")
        .style("font-size", "12px")
        .text("Phase Space");
}

function updatePhaseChart() {
    if (!phaseChart.svg) return;
    
    // Find system with at least 2 state variables
    const trajectories = [];
    
    Object.entries(state.systemStates).forEach(([sysName, sysStates]) => {
        if (sysStates.length > 0 && 
            sysStates[0].state && 
            Array.isArray(sysStates[0].state) && 
            sysStates[0].state.length >= 2) {
            
            const points = sysStates.map(data => ({
                x: data.state[0],
                y: data.state[1]
            }));
            
            trajectories.push({
                name: sysName,
                points: points
            });
            
            // Update domain based on data
            const xMin = d3.min(points, d => d.x);
            const xMax = d3.max(points, d => d.x);
            const yMin = d3.min(points, d => d.y);
            const yMax = d3.max(points, d => d.y);
            
            // Add some padding
            const xPadding = (xMax - xMin) * 0.1;
            const yPadding = (yMax - yMin) * 0.1;
            
            phaseChart.x.domain([xMin - xPadding, xMax + xPadding]);
            phaseChart.y.domain([yMin - yPadding, yMax + yPadding]);
        }
    });
    
    // Update axes
    phaseChart.svg.select(".x-axis")
        .call(d3.axisBottom(phaseChart.x));
    
    phaseChart.svg.select(".y-axis")
        .call(d3.axisLeft(phaseChart.y));
    
    // Update trajectories
    trajectories.forEach((traj, i) => {
        // Create line
        const line = d3.line()
            .x(d => phaseChart.x(d.x))
            .y(d => phaseChart.y(d.y));
        
        // Check if line exists
        let trajLine = phaseChart.svg.select(`.traj-${traj.name}`);
        
        if (trajLine.empty()) {
            // Create new line
            trajLine = phaseChart.svg.append("path")
                .attr("class", `traj-${traj.name}`)
                .style("fill", "none")
                .style("stroke", colors[i % colors.length])
                .style("stroke-width", 2);
        }
        
        // Update line
        if (traj.points.length > 0) {
            trajLine.datum(traj.points)
                .attr("d", line);
            
            // Add current point
            const current = traj.points[traj.points.length - 1];
            
            let currentPoint = phaseChart.svg.select(`.point-${traj.name}`);
            
            if (currentPoint.empty()) {
                currentPoint = phaseChart.svg.append("circle")
                    .attr("class", `point-${traj.name}`)
                    .attr("r", 5)
                    .style("fill", colors[i % colors.length]);
            }
            
            currentPoint
                .attr("cx", phaseChart.x(current.x))
                .attr("cy", phaseChart.y(current.y));
        }
    });
}

// Setup Metrics
function setupMetrics(data) {
    const metricsContainer = d3.select("#metrics");
    
    // Create metric cards
    Object.entries(data.stability_guards).forEach(([name, guard]) => {
        const card = metricsContainer.append("div")
            .attr("class", "col-md-3")
            .append("div")
            .attr("class", "metric-card");
        
        card.append("div")
            .attr("class", "metric-value")
            .attr("id", `metric-${name}`)
            .text(guard.violations);
        
        card.append("div")
            .attr("class", "metric-title")
            .text(`${name} Violations`);
    });
    
    // Add Lyapunov value metrics
    Object.keys(data.lyapunov_functions).forEach(name => {
        const card = metricsContainer.append("div")
            .attr("class", "col-md-3")
            .append("div")
            .attr("class", "metric-card");
        
        card.append("div")
            .attr("class", "metric-value")
            .attr("id", `metric-lyap-${name}`)
            .text("0.0");
        
        card.append("div")
            .attr("class", "metric-title")
            .text(`${name} Value`);
    });
}

function updateMetrics() {
    // Update violation metrics
    Object.entries(state.stabilityViolations).forEach(([name, values]) => {
        if (values.length > 0) {
            const metric = d3.select(`#metric-${name}`);
            if (!metric.empty()) {
                const violations = values[values.length - 1].count;
                metric.text(violations);
                
                // Add alert class if violations > 0
                if (violations > 0) {
                    metric.parent().classed("violation-alert", true);
                    metric.parent().classed("stable-alert", false);
                } else {
                    metric.parent().classed("violation-alert", false);
                    metric.parent().classed("stable-alert", true);
                }
            }
        }
    });
    
    // Update Lyapunov metrics
    Object.entries(state.lyapunovValues).forEach(([name, values]) => {
        if (values.length > 0) {
            const metric = d3.select(`#metric-lyap-${name}`);
            if (!metric.empty()) {
                const value = values[values.length - 1].value.toFixed(3);
                metric.text(value);
            }
        }
    });
}
