import { hologramHint } from './stores/audio';
import ghostEngine from './ghostEngine';

// Subscribe to hologram hints and update renderer
export function connectHologramToAudio() {
  hologramHint.subscribe(hint => {
    // Update any holographic objects that respond to audio
    ghostEngine.scene.traverse(node => {
      if (node.audioResponsive && node.mesh?.material) {
        // Update material uniforms
        const material = node.mesh.material;
        
        if (material.uniforms) {
          // Update hue
          if (material.uniforms.uHue) {
            material.uniforms.uHue.value = hint.hue / 360;
          }
          
          // Update intensity
          if (material.uniforms.uIntensity) {
            material.uniforms.uIntensity.value = hint.intensity;
          }
          
          // Update psi phase
          if (material.uniforms.uPsi) {
            material.uniforms.uPsi.value = hint.psi;
          }
        }
        
        // Update node properties based on audio
        if (node.audioMapping) {
          // Scale based on intensity
          if (node.audioMapping.scaleIntensity) {
            const scale = 1 + hint.intensity * node.audioMapping.scaleIntensity;
            node.transform.setScale(scale);
          }
          
          // Rotate based on psi
          if (node.audioMapping.rotatePsi) {
            const rotation = hint.psi * Math.PI * 2 * node.audioMapping.rotatePsi;
            node.transform.setRotation(0, rotation, 0);
          }
          
          // Color based on hue
          if (node.audioMapping.colorHue && node.mesh.material.setColor) {
            node.mesh.material.setColor({
              h: hint.hue,
              s: 70,
              l: 30 + hint.intensity * 40
            });
          }
        }
      }
    });
  });
}

// Call this after ghost engine is initialized
export function initializeAudioHologram() {
  connectHologramToAudio();
  
  // Create an audio-responsive holographic object
  ghostEngine.addHolographicObject({
    id: 'audio_visualizer',
    geometry: { type: 'sphere', radius: 1, segments: 32 },
    material: {
      type: 'holographic',
      uniforms: {
        uHue: { value: 0 },
        uIntensity: { value: 0 },
        uPsi: { value: 0 }
      }
    },
    position: { x: 0, y: 0, z: 0 },
    audioResponsive: true,
    audioMapping: {
      scaleIntensity: 0.5,
      rotatePsi: 1.0,
      colorHue: true
    }
  });
}