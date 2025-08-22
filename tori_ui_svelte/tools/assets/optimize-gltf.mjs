import { Document, NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS } from '@gltf-transform/extensions';
import {
  dedup,
  weld,
  reorder,
  quantize,
  resample,
  draco,
  meshopt,
  prune,
  textureCompress
} from '@gltf-transform/functions';
import draco3d from 'draco3dgltf';
import { MeshoptEncoder } from 'meshoptimizer';

const [,, inPath, outPath, mode = 'draco'] = process.argv;

if (!inPath || !outPath) {
  console.error('Usage: node tools/assets/optimize-gltf.mjs <in.gltf|.glb> <out.glb> [draco|meshopt]');
  process.exit(1);
}

console.log(`Optimizing ${inPath} -> ${outPath} (mode: ${mode})`);

// Create I/O handler with all extensions
const io = new NodeIO()
  .registerExtensions(ALL_EXTENSIONS)
  .registerDependencies({
    'draco3d.decoder': await draco3d.createDecoderModule(),
    'draco3d.encoder': await draco3d.createEncoderModule(),
  });

try {
  // Read the document
  const document = await io.read(inPath);
  
  // Apply optimizations
  await document.transform(
    // Clean up redundant data
    dedup(),
    
    // Merge nearby vertices
    weld({ tolerance: 0.0001 }),
    
    // Prune unused nodes
    prune(),
    
    // Optimize draw order
    reorder({ encoder: MeshoptEncoder, target: 'size' }),
    
    // Quantize attributes for smaller size
    quantize({
      quantizePosition: 14,
      quantizeNormal: 10,
      quantizeTexcoord: 12,
      quantizeColor: 8,
      quantizeWeight: 8
    }),
    
    // Resample animations if present
    resample()
  );

  // Apply compression based on mode
  if (mode === 'meshopt') {
    // Meshopt compression
    await MeshoptEncoder.ready;
    await document.transform(
      meshopt({ encoder: MeshoptEncoder })
    );
    console.log('Applied Meshopt compression');
  } else {
    // Draco compression (default)
    await document.transform(
      draco({
        method: 'edgebreaker',
        encodeSpeed: 5,
        decodeSpeed: 5,
        quantizePosition: 14,
        quantizeNormal: 10,
        quantizeTexcoord: 12,
        quantizeColor: 8,
        quantizeGeneric: 12
      })
    );
    console.log('Applied Draco compression');
  }

  // Write optimized file
  await io.write(outPath, document);
  
  // Report size reduction
  const fs = await import('fs');
  const inSize = fs.statSync(inPath).size;
  const outSize = fs.statSync(outPath).size;
  const reduction = ((1 - outSize/inSize) * 100).toFixed(1);
  
  console.log(`Success! Size reduced by ${reduction}%`);
  console.log(`  Original: ${(inSize/1024/1024).toFixed(2)} MB`);
  console.log(`  Optimized: ${(outSize/1024/1024).toFixed(2)} MB`);
  
} catch (error) {
  console.error('Optimization failed:', error);
  process.exit(1);
}