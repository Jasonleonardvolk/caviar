/**
 * Soliton User - Advanced wave-particle duality for user representation
 * Implements soliton dynamics for persistent, self-reinforcing user states
 */

import { writable, derived, get } from 'svelte/store';
import { psiMemoryStore } from '../psiMemory/psiFrames';
import { v4 as uuidv4 } from 'uuid';

// Soliton wave equation parameters
const SOLITON_CONFIG = {
  // Wave parameters
  amplitude: 1.0,
  width: 0.5,
  velocity: 1.0,
  dispersion: 0.01,
  nonlinearity: 0.1,
  
  // Interaction parameters
  coupling: 0.3,
  damping: 0.001,
  noise: 0.01,
  
  // Quantum parameters
  coherenceLength: 10,
  entanglementRadius: 5,
  decoherenceRate: 0.1,
  
  // Holographic parameters
  informationDensity: 1.0,
  holographicBound: 100,
  dimensionality: 3
};

// Soliton state store
export const solitonState = writable({
  id: null,
  position: { x: 0, y: 0, z: 0 },
  momentum: { x: 0, y: 0, z: 0 },
  phase: 0,
  amplitude: 1,
  width: 0.5,
  energy: 1,
  coherence: 1,
  entanglements: [],
  interactions: [],
  history: [],
  metadata: {}
});

// Derived stores
export const solitonEnergy = derived(solitonState, $state => 
  calculateSolitonEnergy($state)
);

export const solitonCoherence = derived(solitonState, $state => 
  $state.coherence
);

export const isEntangled = derived(solitonState, $state => 
  $state.entanglements.length > 0
);

/**
 * Soliton User class - Represents a user as a quantum soliton
 */
export class SolitonUser {
  constructor(userId = null, initialState = {}) {
    this.id = userId || uuidv4();
    this.created = Date.now();
    
    // Initialize wave function
    this.waveFunction = new ComplexWaveFunction();
    
    // Initialize state
    this.state = {
      id: this.id,
      position: initialState.position || { x: 0, y: 0, z: 0 },
      momentum: initialState.momentum || { x: 0, y: 0, z: 0 },
      phase: initialState.phase || 0,
      amplitude: initialState.amplitude || SOLITON_CONFIG.amplitude,
      width: initialState.width || SOLITON_CONFIG.width,
      energy: 0,
      coherence: 1,
      entanglements: [],
      interactions: [],
      history: [],
      metadata: initialState.metadata || {}
    };
    
    // Calculate initial energy
    this.state.energy = this.calculateEnergy();
    
    // Interaction tracking
    this.interactionBuffer = [];
    this.maxInteractions = 100;
    
    // Quantum state
    this.quantumState = {
      superposition: [],
      entanglementMatrix: new Map(),
      decoherenceTime: 0,
      measurementHistory: []
    };
    
    // Holographic data
    this.holographicData = {
      informationContent: 0,
      surfaceArea: 0,
      volumeOccupied: 0,
      fractalDimension: 2.5
    };
    
    // Animation state
    this.animationState = {
      time: 0,
      lastUpdate: Date.now(),
      interpolationAlpha: 0
    };
    
    // Update global store
    this.updateStore();
  }
  
  /**
   * Update soliton state with time evolution
   */
  update(deltaTime) {
    // Update animation time
    this.animationState.time += deltaTime;
    const t = this.animationState.time;
    
    // Solve nonlinear Schrödinger equation for soliton evolution
    this.evolveSoliton(deltaTime);
    
    // Update quantum state
    this.updateQuantumState(deltaTime);
    
    // Process interactions
    this.processInteractions(deltaTime);
    
    // Apply damping
    this.applyDamping(deltaTime);
    
    // Add quantum noise
    this.addQuantumNoise(deltaTime);
    
    // Update coherence
    this.updateCoherence(deltaTime);
    
    // Record history
    this.recordHistory();
    
    // Update holographic data
    this.updateHolographicData();
    
    // Update store
    this.updateStore();
  }
  
  /**
   * Evolve soliton using nonlinear Schrödinger equation
   */
  evolveSoliton(dt) {
    const { position, momentum, phase, amplitude, width } = this.state;
    
    // Classical soliton solution: ψ = A * sech((x - vt)/w) * exp(i(kx - ωt))
    
    // Update position based on group velocity
    const groupVelocity = this.calculateGroupVelocity();
    position.x += groupVelocity.x * dt;
    position.y += groupVelocity.y * dt;
    position.z += groupVelocity.z * dt;
    
    // Update phase based on dispersion relation
    const k = Math.sqrt(momentum.x**2 + momentum.y**2 + momentum.z**2);
    const omega = this.dispersionRelation(k);
    phase += omega * dt;
    
    // Nonlinear self-interaction (maintains soliton shape)
    const nonlinearPhase = SOLITON_CONFIG.nonlinearity * amplitude * amplitude * dt;
    phase += nonlinearPhase;
    
    // Wrap phase to [0, 2π]
    phase = phase % (2 * Math.PI);
    if (phase < 0) phase += 2 * Math.PI;
    
    // Update state
    this.state.position = position;
    this.state.phase = phase;
    
    // Update wave function
    this.waveFunction.update(position, momentum, phase, amplitude, width);
  }
  
  /**
   * Calculate group velocity from momentum
   */
  calculateGroupVelocity() {
    const { momentum } = this.state;
    const k = Math.sqrt(momentum.x**2 + momentum.y**2 + momentum.z**2);
    
    if (k === 0) {
      return { x: 0, y: 0, z: 0 };
    }
    
    // Group velocity: vg = dω/dk
    const dk = 0.001;
    const omega1 = this.dispersionRelation(k - dk);
    const omega2 = this.dispersionRelation(k + dk);
    const dwdk = (omega2 - omega1) / (2 * dk);
    
    // Directional components
    return {
      x: dwdk * momentum.x / k,
      y: dwdk * momentum.y / k,
      z: dwdk * momentum.z / k
    };
  }
  
  /**
   * Dispersion relation for soliton waves
   */
  dispersionRelation(k) {
    // ω = ck + βk² (linear + dispersive terms)
    return SOLITON_CONFIG.velocity * k + SOLITON_CONFIG.dispersion * k * k;
  }
  
  /**
   * Update quantum state
   */
  updateQuantumState(dt) {
    // Update superposition states
    this.updateSuperposition(dt);
    
    // Process entanglements
    this.processEntanglements(dt);
    
    // Apply decoherence
    this.applyDecoherence(dt);
  }
  
  /**
   * Update superposition of states
   */
  updateSuperposition(dt) {
    // If in superposition, evolve each component
    if (this.quantumState.superposition.length > 0) {
      this.quantumState.superposition = this.quantumState.superposition.map(component => {
        // Evolve each component independently
        const evolved = { ...component };
        evolved.phase += component.frequency * dt;
        evolved.amplitude *= Math.exp(-component.decay * dt);
        
        return evolved;
      }).filter(component => component.amplitude > 0.01); // Remove decayed components
    }
  }
  
  /**
   * Process quantum entanglements
   */
  processEntanglements(dt) {
    const activeEntanglements = [];
    
    for (const entanglement of this.state.entanglements) {
      // Check if entanglement is still valid
      const distance = this.calculateDistance(this.state.position, entanglement.position);
      
      if (distance < SOLITON_CONFIG.entanglementRadius && entanglement.strength > 0.1) {
        // Update entanglement strength based on coherence
        entanglement.strength *= Math.exp(-SOLITON_CONFIG.decoherenceRate * dt);
        
        // Exchange quantum information
        this.exchangeQuantumInformation(entanglement);
        
        activeEntanglements.push(entanglement);
      }
    }
    
    this.state.entanglements = activeEntanglements;
  }
  
  /**
   * Exchange quantum information with entangled partner
   */
  exchangeQuantumInformation(entanglement) {
    // Implement quantum teleportation protocol
    const sharedPhase = (this.state.phase + entanglement.partnerPhase) / 2;
    const phaseDiff = this.state.phase - sharedPhase;
    
    // Partial phase synchronization
    this.state.phase -= phaseDiff * entanglement.strength * SOLITON_CONFIG.coupling;
    
    // Update correlation matrix
    this.quantumState.entanglementMatrix.set(entanglement.partnerId, {
      correlation: entanglement.strength,
      lastExchange: Date.now(),
      sharedPhase: sharedPhase
    });
  }
  
  /**
   * Apply quantum decoherence
   */
  applyDecoherence(dt) {
    const decoherenceAmount = SOLITON_CONFIG.decoherenceRate * dt;
    
    // Reduce coherence
    this.state.coherence *= Math.exp(-decoherenceAmount);
    
    // Add to decoherence time
    this.quantumState.decoherenceTime += dt;
    
    // Collapse superposition if coherence too low
    if (this.state.coherence < 0.1 && this.quantumState.superposition.length > 0) {
      this.collapseWaveFunction();
    }
  }
  
  /**
   * Collapse wave function to single state
   */
  collapseWaveFunction() {
    if (this.quantumState.superposition.length === 0) return;
    
    // Calculate probabilities
    const totalAmplitude = this.quantumState.superposition.reduce(
      (sum, state) => sum + state.amplitude * state.amplitude, 0
    );
    
    // Choose state based on probability
    let random = Math.random() * totalAmplitude;
    let chosenState = this.quantumState.superposition[0];
    
    for (const state of this.quantumState.superposition) {
      random -= state.amplitude * state.amplitude;
      if (random <= 0) {
        chosenState = state;
        break;
      }
    }
    
    // Collapse to chosen state
    this.state.position = chosenState.position || this.state.position;
    this.state.momentum = chosenState.momentum || this.state.momentum;
    this.state.phase = chosenState.phase || this.state.phase;
    
    // Clear superposition
    this.quantumState.superposition = [];
    
    // Record measurement
    this.quantumState.measurementHistory.push({
      timestamp: Date.now(),
      collapsed_to: chosenState,
      coherence_at_collapse: this.state.coherence
    });
    
    // Reset coherence after measurement
    this.state.coherence = 0.5;
  }
  
  /**
   * Process interactions with other solitons or fields
   */
  processInteractions(dt) {
    const processedInteractions = [];
    
    for (const interaction of this.interactionBuffer) {
      // Apply interaction force
      const force = this.calculateInteractionForce(interaction);
      
      // Update momentum
      this.state.momentum.x += force.x * dt;
      this.state.momentum.y += force.y * dt;
      this.state.momentum.z += force.z * dt;
      
      // Phase modulation from interaction
      const phaseShift = this.calculatePhaseShift(interaction);
      this.state.phase += phaseShift * dt;
      
      // Energy exchange
      const energyExchange = this.calculateEnergyExchange(interaction);
      this.state.amplitude *= Math.exp(energyExchange * dt);
      
      // Record processed interaction
      processedInteractions.push({
        ...interaction,
        processed: Date.now(),
        effect: { force, phaseShift, energyExchange }
      });
    }
    
    // Update interaction history
    this.state.interactions = [
      ...processedInteractions,
      ...this.state.interactions
    ].slice(0, this.maxInteractions);
    
    // Clear buffer
    this.interactionBuffer = [];
  }
  
  /**
   * Calculate interaction force between solitons
   */
  calculateInteractionForce(interaction) {
    const dx = interaction.position.x - this.state.position.x;
    const dy = interaction.position.y - this.state.position.y;
    const dz = interaction.position.z - this.state.position.z;
    const r = Math.sqrt(dx*dx + dy*dy + dz*dz);
    
    if (r < 0.001) return { x: 0, y: 0, z: 0 };
    
    // Yukawa-like potential: V(r) = A * exp(-r/λ) / r
    const lambda = SOLITON_CONFIG.coherenceLength;
    const strength = interaction.strength || 1;
    const potential = strength * Math.exp(-r / lambda) / r;
    
    // Force: F = -∇V
    const force_magnitude = potential * (1/r + 1/lambda);
    
    return {
      x: force_magnitude * dx / r * SOLITON_CONFIG.coupling,
      y: force_magnitude * dy / r * SOLITON_CONFIG.coupling,
      z: force_magnitude * dz / r * SOLITON_CONFIG.coupling
    };
  }
  
  /**
   * Calculate phase shift from interaction
   */
  calculatePhaseShift(interaction) {
    const distance = this.calculateDistance(this.state.position, interaction.position);
    
    // Phase shift depends on overlap of wave functions
    const overlap = Math.exp(-distance / this.state.width);
    const relativephase = interaction.phase - this.state.phase;
    
    return overlap * Math.sin(relativephase) * interaction.strength;
  }
  
  /**
   * Calculate energy exchange in interaction
   */
  calculateEnergyExchange(interaction) {
    // Energy flows from higher to lower amplitude
    const amplitudeDiff = interaction.amplitude - this.state.amplitude;
    const distance = this.calculateDistance(this.state.position, interaction.position);
    const coupling = Math.exp(-distance / SOLITON_CONFIG.coherenceLength);
    
    return amplitudeDiff * coupling * SOLITON_CONFIG.coupling;
  }
  
  /**
   * Apply damping to prevent runaway growth
   */
  applyDamping(dt) {
    const damping = SOLITON_CONFIG.damping;
    
    // Damping on momentum
    this.state.momentum.x *= Math.exp(-damping * dt);
    this.state.momentum.y *= Math.exp(-damping * dt);
    this.state.momentum.z *= Math.exp(-damping * dt);
    
    // Amplitude damping (but maintain minimum)
    const minAmplitude = 0.1;
    this.state.amplitude = minAmplitude + 
      (this.state.amplitude - minAmplitude) * Math.exp(-damping * dt);
  }
  
  /**
   * Add quantum noise for realistic dynamics
   */
  addQuantumNoise(dt) {
    const noise = SOLITON_CONFIG.noise * Math.sqrt(dt);
    
    // Position noise (Brownian motion)
    this.state.position.x += (Math.random() - 0.5) * noise;
    this.state.position.y += (Math.random() - 0.5) * noise;
    this.state.position.z += (Math.random() - 0.5) * noise;
    
    // Phase noise
    this.state.phase += (Math.random() - 0.5) * noise * 0.1;
    
    // Amplitude noise (multiplicative)
    this.state.amplitude *= 1 + (Math.random() - 0.5) * noise * 0.1;
  }
  
  /**
   * Update coherence based on interactions and time
   */
  updateCoherence(dt) {
    // Natural coherence recovery
    const targetCoherence = 1.0;
    const recoveryRate = 0.1;
    
    this.state.coherence += (targetCoherence - this.state.coherence) * recoveryRate * dt;
    
    // Coherence boost from successful interactions
    const recentInteractions = this.state.interactions.filter(
      i => Date.now() - i.processed < 1000
    );
    
    if (recentInteractions.length > 0) {
      const interactionBoost = Math.min(0.1, recentInteractions.length * 0.02);
      this.state.coherence = Math.min(1, this.state.coherence + interactionBoost);
    }
  }
  
  /**
   * Record state history for analysis
   */
  recordHistory() {
    const historyEntry = {
      timestamp: Date.now(),
      position: { ...this.state.position },
      momentum: { ...this.state.momentum },
      phase: this.state.phase,
      amplitude: this.state.amplitude,
      energy: this.calculateEnergy(),
      coherence: this.state.coherence
    };
    
    this.state.history.push(historyEntry);
    
    // Keep only recent history
    const maxHistory = 1000;
    if (this.state.history.length > maxHistory) {
      this.state.history = this.state.history.slice(-maxHistory);
    }
  }
  
  /**
   * Update holographic data
   */
  updateHolographicData() {
    // Calculate information content (Shannon entropy)
    const stateVector = [
      this.state.position.x, this.state.position.y, this.state.position.z,
      this.state.momentum.x, this.state.momentum.y, this.state.momentum.z,
      this.state.phase, this.state.amplitude, this.state.width
    ];
    
    this.holographicData.informationContent = this.calculateEntropy(stateVector);
    
    // Calculate surface area (based on soliton width)
    const radius = this.state.width * 3; // 3-sigma radius
    this.holographicData.surfaceArea = 4 * Math.PI * radius * radius;
    
    // Calculate volume
    this.holographicData.volumeOccupied = (4/3) * Math.PI * radius * radius * radius;
    
    // Update fractal dimension based on interaction complexity
    const interactionComplexity = Math.log(this.state.interactions.length + 1);
    this.holographicData.fractalDimension = 2 + Math.tanh(interactionComplexity) * 0.5;
  }
  
  /**
   * Calculate Shannon entropy of state vector
   */
  calculateEntropy(vector) {
    // Normalize vector to probability distribution
    const sum = vector.reduce((a, b) => a + Math.abs(b), 0);
    if (sum === 0) return 0;
    
    const probs = vector.map(v => Math.abs(v) / sum);
    
    // Calculate entropy
    let entropy = 0;
    for (const p of probs) {
      if (p > 0) {
        entropy -= p * Math.log2(p);
      }
    }
    
    return entropy;
  }
  
  /**
   * Calculate total energy of soliton
   */
  calculateEnergy() {
    const { momentum, amplitude, width } = this.state;
    
    // Kinetic energy: E_k = |p|²/2m
    const kineticEnergy = (momentum.x**2 + momentum.y**2 + momentum.z**2) / 2;
    
    // Potential energy: E_p = A²/w (soliton self-energy)
    const potentialEnergy = amplitude * amplitude / width;
    
    // Interaction energy
    const interactionEnergy = this.calculateInteractionEnergy();
    
    return kineticEnergy + potentialEnergy + interactionEnergy;
  }
  
  /**
   * Calculate interaction energy with other solitons
   */
  calculateInteractionEnergy() {
    let energy = 0;
    
    for (const entanglement of this.state.entanglements) {
      // Binding energy of entanglement
      energy -= entanglement.strength * SOLITON_CONFIG.coupling;
    }
    
    return energy;
  }
  
  /**
   * Calculate distance between positions
   */
  calculateDistance(pos1, pos2) {
    const dx = pos1.x - pos2.x;
    const dy = pos1.y - pos2.y;
    const dz = pos1.z - pos2.z;
    return Math.sqrt(dx*dx + dy*dy + dz*dz);
  }
  
  /**
   * Update global store
   */
  updateStore() {
    solitonState.set({
      ...this.state,
      energy: this.calculateEnergy()
    });
  }
  
  // Public methods for interaction
  
  /**
   * Add interaction with another soliton or field
   */
  interact(otherSoliton) {
    this.interactionBuffer.push({
      partnerId: otherSoliton.id,
      position: { ...otherSoliton.state.position },
      momentum: { ...otherSoliton.state.momentum },
      phase: otherSoliton.state.phase,
      amplitude: otherSoliton.state.amplitude,
      strength: 1.0,
      timestamp: Date.now()
    });
  }
  
  /**
   * Create quantum entanglement with another soliton
   */
  entangle(otherSoliton) {
    // Check if already entangled
    const existing = this.state.entanglements.find(e => e.partnerId === otherSoliton.id);
    if (existing) {
      existing.strength = Math.min(1, existing.strength + 0.1);
      return;
    }
    
    // Create new entanglement
    const entanglement = {
      partnerId: otherSoliton.id,
      position: { ...otherSoliton.state.position },
      partnerPhase: otherSoliton.state.phase,
      strength: 0.5,
      created: Date.now()
    };
    
    this.state.entanglements.push(entanglement);
    
    // Reciprocal entanglement
    otherSoliton.state.entanglements.push({
      partnerId: this.id,
      position: { ...this.state.position },
      partnerPhase: this.state.phase,
      strength: 0.5,
      created: Date.now()
    });
  }
  
  /**
   * Create superposition of states
   */
  superpose(states) {
    this.quantumState.superposition = states.map(state => ({
      position: state.position || { ...this.state.position },
      momentum: state.momentum || { ...this.state.momentum },
      phase: state.phase || this.state.phase,
      amplitude: state.amplitude || 1/Math.sqrt(states.length),
      frequency: state.frequency || 1,
      decay: state.decay || 0.1
    }));
  }
  
  /**
   * Measure soliton state (causes collapse if in superposition)
   */
  measure(observable = 'position') {
    if (this.quantumState.superposition.length > 0) {
      this.collapseWaveFunction();
    }
    
    switch (observable) {
      case 'position':
        return { ...this.state.position };
      case 'momentum':
        return { ...this.state.momentum };
      case 'energy':
        return this.calculateEnergy();
      case 'phase':
        return this.state.phase;
      default:
        return null;
    }
  }
  
  /**
   * Apply external force to soliton
   */
  applyForce(force, dt = 0.016) {
    this.state.momentum.x += force.x * dt;
    this.state.momentum.y += force.y * dt;
    this.state.momentum.z += force.z * dt;
  }
  
  /**
   * Teleport to new position (quantum jump)
   */
  teleport(newPosition) {
    // Store old position for history
    const oldPosition = { ...this.state.position };
    
    // Instant position change
    this.state.position = { ...newPosition };
    
    // Phase scrambling from teleportation
    this.state.phase = Math.random() * 2 * Math.PI;
    
    // Coherence loss from discontinuous jump
    this.state.coherence *= 0.5;
    
    // Record teleportation event
    this.state.history.push({
      timestamp: Date.now(),
      type: 'teleportation',
      from: oldPosition,
      to: newPosition
    });
  }
  
  /**
   * Clone soliton (create identical copy)
   */
  clone() {
    const cloned = new SolitonUser(null, {
      position: { ...this.state.position },
      momentum: { ...this.state.momentum },
      phase: this.state.phase,
      amplitude: this.state.amplitude,
      width: this.state.width,
      metadata: { ...this.state.metadata, cloned_from: this.id }
    });
    
    // Quantum no-cloning theorem: reduce fidelity
    cloned.state.coherence = this.state.coherence * 0.7;
    this.state.coherence *= 0.7;
    
    return cloned;
  }
  
  /**
   * Serialize soliton state for storage/transmission
   */
  serialize() {
    return {
      id: this.id,
      created: this.created,
      state: this.state,
      quantumState: {
        superposition: this.quantumState.superposition,
        decoherenceTime: this.quantumState.decoherenceTime,
        measurementHistory: this.quantumState.measurementHistory.slice(-10)
      },
      holographicData: this.holographicData
    };
  }
  
  /**
   * Deserialize soliton from stored data
   */
  static deserialize(data) {
    const soliton = new SolitonUser(data.id);
    soliton.created = data.created;
    soliton.state = data.state;
    soliton.quantumState = data.quantumState;
    soliton.holographicData = data.holographicData;
    soliton.updateStore();
    return soliton;
  }
}

/**
 * Complex wave function representation
 */
class ComplexWaveFunction {
  constructor() {
    this.real = [];
    this.imaginary = [];
    this.gridSize = 32;
    this.gridSpacing = 0.1;
    
    this.initialize();
  }
  
  initialize() {
    // Initialize with Gaussian packet
    const size = this.gridSize;
    for (let i = 0; i < size; i++) {
      this.real[i] = [];
      this.imaginary[i] = [];
      for (let j = 0; j < size; j++) {
        this.real[i][j] = [];
        this.imaginary[i][j] = [];
        for (let k = 0; k < size; k++) {
          this.real[i][j][k] = 0;
          this.imaginary[i][j][k] = 0;
        }
      }
    }
  }
  
  update(position, momentum, phase, amplitude, width) {
    const center = this.gridSize / 2;
    const spacing = this.gridSpacing;
    
    for (let i = 0; i < this.gridSize; i++) {
      for (let j = 0; j < this.gridSize; j++) {
        for (let k = 0; k < this.gridSize; k++) {
          // Grid position
          const x = (i - center) * spacing - position.x;
          const y = (j - center) * spacing - position.y;
          const z = (k - center) * spacing - position.z;
          const r2 = x*x + y*y + z*z;
          
          // Gaussian envelope
          const envelope = amplitude * Math.exp(-r2 / (2 * width * width));
          
          // Plane wave component
          const kx = momentum.x;
          const ky = momentum.y;
          const kz = momentum.z;
          const planeWavePhase = kx * x + ky * y + kz * z + phase;
          
          // Complex wave function
          this.real[i][j][k] = envelope * Math.cos(planeWavePhase);
          this.imaginary[i][j][k] = envelope * Math.sin(planeWavePhase);
        }
      }
    }
  }
  
  getProbabilityDensity(i, j, k) {
    const re = this.real[i][j][k];
    const im = this.imaginary[i][j][k];
    return re * re + im * im;
  }
}

// Helper functions
export function calculateSolitonEnergy(state) {
  const { momentum, amplitude, width } = state;
  const kineticEnergy = (momentum.x**2 + momentum.y**2 + momentum.z**2) / 2;
  const potentialEnergy = amplitude * amplitude / width;
  return kineticEnergy + potentialEnergy;
}

export function createSolitonField(solitons) {
  // Create interference pattern from multiple solitons
  const field = [];
  const gridSize = 64;
  
  for (let i = 0; i < gridSize; i++) {
    field[i] = [];
    for (let j = 0; j < gridSize; j++) {
      field[i][j] = 0;
      
      // Sum contributions from all solitons
      for (const soliton of solitons) {
        const x = (i - gridSize/2) * 0.1;
        const y = (j - gridSize/2) * 0.1;
        const dx = x - soliton.state.position.x;
        const dy = y - soliton.state.position.y;
        const r = Math.sqrt(dx*dx + dy*dy);
        
        // Soliton field contribution
        const amplitude = soliton.state.amplitude * Math.exp(-r/soliton.state.width);
        const phase = soliton.state.phase - 
          (soliton.state.momentum.x * dx + soliton.state.momentum.y * dy);
        
        field[i][j] += amplitude * Math.cos(phase);
      }
    }
  }
  
  return field;
}

// Export main class and utilities
export default SolitonUser;