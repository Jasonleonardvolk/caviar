// PLACEHOLDER - Replace with production build before shipping
// Download actual basis_transcoder.js from:
// https://github.com/BinomialLLC/basis_universal/tree/master/webgl/transcoder
// Path must remain stable for SW precache integrity: /wasm/basis_transcoder.js

console.warn('[basis_transcoder.js] PLACEHOLDER FILE - Replace with actual Basis Universal transcoder');

// Minimal stub to prevent errors during development
self.BASIS = {
  initializeBasis: function() {
    console.warn('[BASIS] Using placeholder - transcoding will not work');
  },
  BasisFile: function() {
    return {
      close: function() {},
      getNumImages: function() { return 0; },
      getNumLevels: function() { return 0; },
      getImageWidth: function() { return 256; },
      getImageHeight: function() { return 256; },
      getImageTranscodedSizeInBytes: function() { return 256 * 256 * 4; },
      transcodeImage: function() { return false; }
    };
  },
  KTX2File: function() {
    return {
      close: function() {},
      isValid: function() { return false; },
      getNumImages: function() { return 0; },
      getNumLevels: function() { return 0; },
      getImageWidth: function() { return 256; },
      getImageHeight: function() { return 256; },
      getImageTranscodedSizeInBytes: function() { return 256 * 256 * 4; },
      transcodeImage: function() { return false; }
    };
  },
  TranscoderTextureFormat: {
    RGBA32: 13,
    RGB565: 14,
    RGBA4444: 15,
    ETC1_RGB: 0,
    BC7_M5: 6,
    ASTC_4x4: 10
  }
};
