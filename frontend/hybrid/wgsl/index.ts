// Vite-style raw shader imports so callers do: import { lightFieldComposerWGSL } from '.../wgsl';
//
// If you already use a shader bundler, adapt the suffix (`?raw`) to your setup.

import lightFieldComposerWGSL from "./lightFieldComposer.wgsl?raw";
import lightFieldComposerEnhancedWGSL from "./lightFieldComposerEnhanced.wgsl?raw";
import hybridWavefieldBlendWGSL from "../hybridWavefieldBlend.wgsl?raw";
import multiDepthWaveSynthWGSL from "../multiDepthWaveSynth.wgsl?raw";
import phaseOcclusionWGSL from "../phaseOcclusion.wgsl?raw";

export {
  lightFieldComposerWGSL,
  lightFieldComposerEnhancedWGSL,
  hybridWavefieldBlendWGSL,
  multiDepthWaveSynthWGSL,
  phaseOcclusionWGSL,
};