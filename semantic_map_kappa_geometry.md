# ALAN IDE: Concept Field Canvas – κ-Geometry Layout Integration (2025-05-07)

## Data Contract (JSON)

```
{
  "nodes": [
    {
      "id": "concept_001",
      "label": "Quantum Entanglement",
      "phase": 0.72,
      "resonance": 0.48,
      "entropy": 0.21,
      "usage": 0.93,
      "x": null,
      "y": null,
      "group": "cluster_1"
    },
    // ...
  ],
  "edges": [
    {
      "id": "edge_001",
      "source": "concept_001",
      "target": "concept_002",
      "coupling": 0.72,
      "weight": 0.72
    }
    // ...
  ],
  "meta": {
    "timestamp": "2025-05-07T12:32:26-05:00",
    "graph_id": "alan_v2_main"
  }
}
```

---

## React Scaffold (`ConceptFieldCanvas.jsx`)

```jsx
import React, { useEffect, useState } from "react";
// You can use D3-force for layout, or custom κ-geometry logic

export default function ConceptFieldCanvas() {
  const [graph, setGraph] = useState({ nodes: [], edges: [] });

  useEffect(() => {
    // TODO: Replace with your backend API endpoint
    fetch("/api/concept-graph")
      .then(res => res.json())
      .then(setGraph);
  }, []);

  // Layout logic (κ-geometry) will be added here

  return (
    <svg width="100%" height="100%">
      {/* Render edges */}
      {graph.edges.map(edge => (
        <line
          key={edge.id}
          x1={getNodeX(edge.source)}
          y1={getNodeY(edge.source)}
          x2={getNodeX(edge.target)}
          y2={getNodeY(edge.target)}
          stroke="#A9B1BD"
          strokeWidth={edge.weight * 4}
          opacity={0.6}
        />
      ))}
      {/* Render nodes */}
      {graph.nodes.map(node => (
        <circle
          key={node.id}
          cx={node.x}
          cy={node.y}
          r={18 + node.usage * 10}
          fill={phaseToColor(node.phase)}
          stroke="#00FFCC"
          strokeWidth={2}
        >
          <title>{node.label}</title>
        </circle>
      ))}
    </svg>
  );

  // Helper functions (to be implemented)
  function getNodeX(id) {
    const node = graph.nodes.find(n => n.id === id);
    return node && node.x !== null ? node.x : 200; // fallback position
  }
  function getNodeY(id) {
    const node = graph.nodes.find(n => n.id === id);
    return node && node.y !== null ? node.y : 200; // fallback position
  }
  function phaseToColor(phase) {
    // HSV to RGB for phase coloring
    const hue = phase * 360;
    return `hsl(${hue}, 100%, 60%)`;
  }
}
```

---

## κ-Geometry Layout Integration (D3-force + custom)

- Start with `d3-force` for force-directed layout.
- Add custom forces for clustering (group) and phase (radial layout).
- Edge weights/coupling influence spring strength.
- Positions update on simulation tick.

**Sample hook:**

```jsx
import * as d3 from "d3-force";

function useKappaGeometryLayout(nodes, edges, { width, height }) {
  const [positions, setPositions] = React.useState({});

  React.useEffect(() => {
    if (!nodes.length) return;

    // Initialize simulation
    const sim = d3.forceSimulation(nodes)
      .force("charge", d3.forceManyBody().strength(-80))
      .force("link", d3.forceLink(edges).id(d => d.id).distance(120).strength(e => e.weight || 0.5))
      .force("center", d3.forceCenter(width / 2, height / 2))
      // κ-geometry: cluster force
      .force("cluster", clusterForce(nodes))
      // κ-geometry: phase radial force (optional)
      .force("phase", phaseRadialForce(nodes, width, height));

    sim.on("tick", () => {
      setPositions(
        Object.fromEntries(nodes.map(n => [n.id, { x: n.x, y: n.y }]))
      );
    });

    return () => sim.stop();
  }, [nodes, edges, width, height]);

  return positions;
}

// Example custom forces (implement as needed)
function clusterForce(nodes) {
  // Pull nodes toward their cluster centroid
  // ...
}
function phaseRadialForce(nodes, width, height) {
  // Arrange nodes in a circle by phase
  // ...
}
```

---

# All code and design snippets are now saved for future reference and implementation
