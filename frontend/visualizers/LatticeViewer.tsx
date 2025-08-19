import React, { useRef, useEffect, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass';

interface LatticeNode {
  id: string;
  position: [number, number, number];
  value: number;
  phase: number;
}

interface LatticeEdge {
  source: string;
  target: string;
  coupling: number;
}

interface LatticeData {
  nodes: LatticeNode[];
  edges: LatticeEdge[];
  topology: string;
  breathingRatio?: number;
}

export const LatticeViewer: React.FC = () => {
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const composerRef = useRef<EffectComposer | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const frameRef = useRef<number>(0);

  const [latticeData, setLatticeData] = useState<LatticeData | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [showPhase, setShowPhase] = useState(true);
  const [showCouplings, setShowCouplings] = useState(true);

  useEffect(() => {
    if (!mountRef.current) return;

    // Initialize Three.js scene
    const width = mountRef.current.clientWidth;
    const height = mountRef.current.clientHeight;

    // Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000011);
    scene.fog = new THREE.Fog(0x000011, 10, 100);
    sceneRef.current = scene;

    // Camera
    const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    camera.position.set(15, 15, 15);
    cameraRef.current = camera;

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    rendererRef.current = renderer;
    mountRef.current.appendChild(renderer.domElement);

    // Post-processing
    const composer = new EffectComposer(renderer);
    const renderPass = new RenderPass(scene, camera);
    composer.addPass(renderPass);

    const bloomPass = new UnrealBloomPass(
      new THREE.Vector2(width, height),
      0.5,  // bloom strength
      0.4,  // radius
      0.85  // threshold
    );
    composer.addPass(bloomPass);
    composerRef.current = composer;

    // Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controlsRef.current = controls;

    // Lights
    const ambientLight = new THREE.AmbientLight(0x404040, 0.5);
    scene.add(ambientLight);

    const pointLight = new THREE.PointLight(0xffffff, 1);
    pointLight.position.set(10, 10, 10);
    scene.add(pointLight);

    // Connect to WebSocket for lattice updates
    const ws = new WebSocket('ws://localhost:8767/lattice');
    
    ws.onopen = () => {
      setIsConnected(true);
      console.log('Connected to lattice stream');
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'lattice_update') {
        setLatticeData(data.lattice);
      }
    };

    ws.onclose = () => {
      setIsConnected(false);
    };

    // Animation loop
    const animate = () => {
      frameRef.current = requestAnimationFrame(animate);
      
      if (controlsRef.current) {
        controlsRef.current.update();
      }

      if (composerRef.current) {
        composerRef.current.render();
      }
    };
    animate();

    // Handle resize
    const handleResize = () => {
      if (!mountRef.current || !cameraRef.current || !rendererRef.current || !composerRef.current) return;
      
      const width = mountRef.current.clientWidth;
      const height = mountRef.current.clientHeight;
      
      cameraRef.current.aspect = width / height;
      cameraRef.current.updateProjectionMatrix();
      
      rendererRef.current.setSize(width, height);
      composerRef.current.setSize(width, height);
    };
    window.addEventListener('resize', handleResize);

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(frameRef.current);
      ws.close();
      
      if (mountRef.current && rendererRef.current) {
        mountRef.current.removeChild(rendererRef.current.domElement);
      }
      
      rendererRef.current?.dispose();
    };
  }, []);

  useEffect(() => {
    if (!sceneRef.current || !latticeData) return;

    // Clear existing lattice geometry
    const toRemove: THREE.Object3D[] = [];
    sceneRef.current.traverse((child) => {
      if (child.userData.latticeElement) {
        toRemove.push(child);
      }
    });
    toRemove.forEach(obj => sceneRef.current!.remove(obj));

    // Create node geometry
    const nodeGeometry = new THREE.SphereGeometry(0.3, 32, 16);
    const nodeMaterial = new THREE.MeshPhongMaterial({
      emissive: new THREE.Color(0x00ff88),
      emissiveIntensity: 0.5
    });

    // Add nodes
    const nodeMap = new Map<string, THREE.Mesh>();
    latticeData.nodes.forEach(node => {
      const mesh = new THREE.Mesh(nodeGeometry, nodeMaterial.clone());
      mesh.position.set(...node.position);
      mesh.userData.latticeElement = true;
      mesh.userData.nodeId = node.id;
      
      // Color based on phase if enabled
      if (showPhase) {
        const hue = node.phase / (2 * Math.PI);
        mesh.material.color.setHSL(hue, 1, 0.5);
      }
      
      // Scale based on value
      const scale = 0.5 + Math.abs(node.value) * 2;
      mesh.scale.setScalar(scale);
      
      sceneRef.current!.add(mesh);
      nodeMap.set(node.id, mesh);
    });

    // Add edges if enabled
    if (showCouplings) {
      const lineMaterial = new THREE.LineBasicMaterial({
        color: 0x4444ff,
        transparent: true,
        opacity: 0.6
      });

      latticeData.edges.forEach(edge => {
        const sourceNode = nodeMap.get(edge.source);
        const targetNode = nodeMap.get(edge.target);
        
        if (sourceNode && targetNode) {
          const geometry = new THREE.BufferGeometry().setFromPoints([
            sourceNode.position,
            targetNode.position
          ]);
          
          const line = new THREE.Line(geometry, lineMaterial.clone());
          line.material.opacity = Math.abs(edge.coupling);
          line.userData.latticeElement = true;
          
          sceneRef.current!.add(line);
        }
      });
    }

    // Add topology label
    const topologyLabel = document.getElementById('topology-label');
    if (topologyLabel) {
      topologyLabel.textContent = `Topology: ${latticeData.topology}`;
      if (latticeData.breathingRatio !== undefined) {
        topologyLabel.textContent += ` (breathing ratio: ${latticeData.breathingRatio.toFixed(2)})`;
      }
    }

  }, [latticeData, showPhase, showCouplings]);

  return (
    <div className="lattice-viewer">
      <div className="viewer-controls">
        <h3>Lattice Viewer</h3>
        <div className="control-group">
          <label>
            <input
              type="checkbox"
              checked={showPhase}
              onChange={(e) => setShowPhase(e.target.checked)}
            />
            Show Phase
          </label>
          <label>
            <input
              type="checkbox"
              checked={showCouplings}
              onChange={(e) => setShowCouplings(e.target.checked)}
            />
            Show Couplings
          </label>
        </div>
        <div className="connection-status">
          <span className={`status-dot ${isConnected ? 'connected' : 'disconnected'}`} />
          {isConnected ? 'Connected' : 'Disconnected'}
        </div>
        <div id="topology-label" className="topology-label">
          Topology: Loading...
        </div>
      </div>
      <div ref={mountRef} className="three-container" />

      <style jsx>{`
        .lattice-viewer {
          width: 100%;
          height: 600px;
          position: relative;
          background: #000011;
          border-radius: 8px;
          overflow: hidden;
        }

        .viewer-controls {
          position: absolute;
          top: 20px;
          left: 20px;
          background: rgba(255, 255, 255, 0.9);
          padding: 16px;
          border-radius: 8px;
          z-index: 10;
          min-width: 200px;
        }

        .viewer-controls h3 {
          margin: 0 0 12px 0;
          font-size: 18px;
        }

        .control-group {
          display: flex;
          flex-direction: column;
          gap: 8px;
          margin-bottom: 12px;
        }

        .control-group label {
          display: flex;
          align-items: center;
          gap: 8px;
          cursor: pointer;
        }

        .connection-status {
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 14px;
          margin-bottom: 8px;
        }

        .status-dot {
          width: 10px;
          height: 10px;
          border-radius: 50%;
        }

        .status-dot.connected {
          background-color: #4caf50;
        }

        .status-dot.disconnected {
          background-color: #f44336;
        }

        .topology-label {
          font-size: 14px;
          color: #666;
          font-style: italic;
        }

        .three-container {
          width: 100%;
          height: 100%;
        }
      `}</style>
    </div>
  );
};

export default LatticeViewer;
