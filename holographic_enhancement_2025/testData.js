/**
 * Test Data for Holographic System Testing
 */

// Reference wavefields for comparison
export const referenceWavefields = {
  fft: {
    size: 512,
    data: generateReferenceWavefield('fft', 512)
  },
  penrose: {
    size: 512,
    data: generateReferenceWavefield('penrose', 512)
  },
  hybrid: {
    size: 512,
    data: generateReferenceWavefield('hybrid', 512)
  }
};

// Reference quilts
export const referenceQuilts = {
  standard: {
    width: 3360,
    height: 3360,
    views: 45,
    data: generateReferenceQuilt('standard')
  },
  enhanced: {
    width: 3360,
    height: 3360,
    views: 45,
    data: generateReferenceQuilt('enhanced')
  }
};

// Test concepts
export const testConcepts = [
  {
    id: 'concept-1',
    name: 'Quantum Entanglement',
    description: 'Non-local correlations between particles',
    position: [0, 0, 0],
    category: 'physics',
    hologram: {
      psi_phase: 0,
      phase_coherence: 0.95,
      oscillator_phases: Array(32).fill(0).map((_, i) => i * Math.PI / 16),
      oscillator_frequencies: Array(32).fill(0).map((_, i) => 200 + i * 25),
      dominant_frequency: 440,
      color: [0.2, 0.5, 1.0],
      size: 1.0,
      intensity: 0.8,
      rotation_speed: 0.5
    }
  },
  {
    id: 'concept-2',
    name: 'Neural Network',
    description: 'Interconnected processing nodes',
    position: [2, 0, 0],
    category: 'ai',
    hologram: {
      psi_phase: Math.PI / 4,
      phase_coherence: 0.7,
      oscillator_phases: Array(32).fill(0).map((_, i) => i * Math.PI / 8),
      oscillator_frequencies: Array(32).fill(0).map((_, i) => 300 + i * 30),
      dominant_frequency: 600,
      color: [1.0, 0.3, 0.3],
      size: 0.8,
      intensity: 0.9,
      rotation_speed: 0.8
    }
  },
  {
    id: 'concept-3',
    name: 'Holographic Memory',
    description: 'Distributed information storage',
    position: [-1, 1, 0],
    category: 'memory',
    hologram: {
      psi_phase: Math.PI / 2,
      phase_coherence: 0.85,
      oscillator_phases: Array(32).fill(0).map(() => Math.random() * Math.PI * 2),
      oscillator_frequencies: Array(32).fill(0).map(() => 100 + Math.random() * 500),
      dominant_frequency: 350,
      color: [0.5, 1.0, 0.5],
      size: 1.2,
      intensity: 0.7,
      rotation_speed: 0.3
    }
  },
  {
    id: 'concept-4',
    name: 'Wave Function',
    description: 'Quantum state description',
    position: [0, -1, 1],
    category: 'physics',
    hologram: {
      psi_phase: Math.PI * 3 / 4,
      phase_coherence: 0.99,
      oscillator_phases: Array(32).fill(0).map((_, i) => Math.sin(i / 32 * Math.PI * 2)),
      oscillator_frequencies: Array(32).fill(0).map((_, i) => 400 + 50 * Math.cos(i / 32 * Math.PI)),
      dominant_frequency: 425,
      color: [0.8, 0.4, 0.8],
      size: 0.9,
      intensity: 0.85,
      rotation_speed: 0.6
    }
  },
  {
    id: 'concept-5',
    name: 'Consciousness',
    description: 'Emergent awareness phenomenon',
    position: [1, 1, 1],
    category: 'philosophy',
    hologram: {
      psi_phase: Math.PI,
      phase_coherence: 0.6,
      oscillator_phases: Array(32).fill(0).map(() => Math.random() * Math.PI * 2),
      oscillator_frequencies: Array(32).fill(0).map(() => 50 + Math.random() * 800),
      dominant_frequency: 432,
      color: [0.9, 0.9, 0.1],
      size: 1.5,
      intensity: 1.0,
      rotation_speed: 1.0
    }
  }
];

// Test audio features
export const testAudioFeatures = {
  spectral_centroid: 440,
  spectral_flatness: 0.8,
  pitch: 440,
  band_energies: [
    0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2,
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
    0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2,
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8
  ],
  zero_crossing_rate: 0.1,
  rms_energy: 0.5,
  spectral_rolloff: 2000,
  mfcc: Array(13).fill(0).map(() => Math.random() * 2 - 1)
};

// Test scenes
export const testScenes = {
  simple: {
    concepts: testConcepts.slice(0, 2),
    relations: [
      {
        id: 'rel-1',
        source_id: 'concept-1',
        target_id: 'concept-2',
        type: 'related',
        strength: 0.8,
        hologram: {
          color: [0.5, 0.5, 1.0],
          width: 0.1,
          energy_flow: 0.8,
          particle_count: 30,
          pulse_speed: 1.0,
          wave_frequency: 2,
          wave_amplitude: 0.2,
          phase_offset: 0
        }
      }
    ],
    total_concepts: 2
  },
  complex: {
    concepts: testConcepts,
    relations: generateComplexRelations(testConcepts),
    total_concepts: testConcepts.length
  }
};

// Performance baselines
export const performanceBaselines = {
  fft: {
    '512': { min: 5, max: 15, target: 10 },
    '1024': { min: 20, max: 40, target: 30 },
    '2048': { min: 80, max: 160, target: 120 }
  },
  penrose: {
    '512': { min: 15, max: 30, target: 22 },
    '1024': { min: 60, max: 120, target: 90 },
    '2048': { min: 240, max: 480, target: 360 }
  },
  ai_assisted: {
    '512': { min: 20, max: 40, target: 30 },
    '1024': { min: 80, max: 160, target: 120 },
    '2048': { min: 320, max: 640, target: 480 }
  }
};

// Helper functions
function generateReferenceWavefield(type, size) {
  const data = new Float32Array(size * size * 2); // Complex values
  
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const idx = (y * size + x) * 2;
      const u = x / size - 0.5;
      const v = y / size - 0.5;
      const r = Math.sqrt(u * u + v * v);
      
      switch (type) {
        case 'fft':
          // Sinc pattern typical of FFT
          const sinc = r === 0 ? 1 : Math.sin(r * 10) / (r * 10);
          data[idx] = sinc * Math.cos(r * 20);
          data[idx + 1] = sinc * Math.sin(r * 20);
          break;
          
        case 'penrose':
          // Quasicrystal-like pattern
          const phi = Math.atan2(v, u);
          const penrose = Math.cos(r * 10) * Math.cos(5 * phi);
          data[idx] = penrose;
          data[idx + 1] = penrose * 0.5;
          break;
          
        case 'hybrid':
          // Combination of patterns
          const sinc2 = r === 0 ? 1 : Math.sin(r * 10) / (r * 10);
          const phi2 = Math.atan2(v, u);
          data[idx] = sinc2 * Math.cos(r * 20) * 0.7 + Math.cos(5 * phi2) * 0.3;
          data[idx + 1] = sinc2 * Math.sin(r * 20) * 0.7 + Math.sin(5 * phi2) * 0.3;
          break;
      }
    }
  }
  
  return data;
}

function generateReferenceQuilt(type) {
  // Generate reference quilt data
  const size = 3360;
  const views = 45;
  const viewWidth = 420;
  const viewHeight = 560;
  
  // Mock quilt data - in reality would be actual rendered views
  return {
    type,
    checksum: generateChecksum(type),
    metadata: {
      renderTime: type === 'enhanced' ? 150 : 100,
      quality: type === 'enhanced' ? 'high' : 'normal'
    }
  };
}

function generateComplexRelations(concepts) {
  const relations = [];
  
  // Create a web of relations
  for (let i = 0; i < concepts.length; i++) {
    for (let j = i + 1; j < concepts.length; j++) {
      // 50% chance of relation
      if (Math.random() > 0.5) {
        relations.push({
          id: `rel-${i}-${j}`,
          source_id: concepts[i].id,
          target_id: concepts[j].id,
          type: ['related', 'influences', 'depends_on'][Math.floor(Math.random() * 3)],
          strength: 0.3 + Math.random() * 0.7,
          hologram: {
            color: [Math.random(), Math.random(), Math.random()],
            width: 0.05 + Math.random() * 0.1,
            energy_flow: 0.5 + Math.random() * 0.5,
            particle_count: 10 + Math.floor(Math.random() * 40),
            pulse_speed: 0.5 + Math.random() * 1.5,
            wave_frequency: 1 + Math.random() * 3,
            wave_amplitude: 0.1 + Math.random() * 0.3,
            phase_offset: Math.random() * Math.PI * 2
          }
        });
      }
    }
  }
  
  return relations;
}

function generateChecksum(data) {
  // Simple checksum for testing
  let sum = 0;
  const str = JSON.stringify(data);
  for (let i = 0; i < str.length; i++) {
    sum = ((sum << 5) - sum) + str.charCodeAt(i);
    sum = sum & sum; // Convert to 32-bit integer
  }
  return Math.abs(sum).toString(16);
}

// Export test data
export default {
  referenceWavefields,
  referenceQuilts,
  testConcepts,
  testAudioFeatures,
  testScenes,
  performanceBaselines
};
