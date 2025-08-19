/**
 * Basis Universal transcoder worker
 * Handles KTX2 texture decompression in a Web Worker
 * Requires basis_transcoder.js and basis_transcoder.wasm in /public/wasm/
 */

let basisReady: Promise<any> | null = null;
let BasisModule: any = null;

async function initBasis() {
  if (!basisReady) {
    basisReady = new Promise(async (resolve, reject) => {
      try {
        // @ts-ignore - Global BASIS object
        self.BASIS = {};
        
        // Import the Basis transcoder JavaScript glue
        // It will automatically fetch the .wasm file from the same directory
        importScripts('/wasm/basis_transcoder.js');
        
        // @ts-ignore - BASIS is now available
        const factory = (self as any).BASIS;
        
        // Initialize the module
        const module = await factory({});
        BasisModule = module;
        
        // Initialize Basis
        if (module.initializeBasis) {
          module.initializeBasis();
        }
        
        resolve(module);
      } catch (error) {
        console.error('[BasisWorker] Initialization failed:', error);
        reject(error);
      }
    });
  }
  return basisReady;
}

export type TranscodeFormat = 'astc' | 'bc7' | 'etc2' | 'rgba';

export interface TranscodeMessage {
  id: number;
  type: 'init' | 'transcode' | 'dispose';
  data?: ArrayBuffer;
  format?: TranscodeFormat;
}

export interface TranscodeResult {
  id: number;
  ok: boolean;
  error?: string;
  levels?: {
    width: number;
    height: number;
    data: ArrayBuffer;
  }[];
}

// Handle messages from main thread
self.onmessage = async (event: MessageEvent<TranscodeMessage>) => {
  const msg = event.data;
  
  switch (msg.type) {
    case 'init':
      try {
        await initBasis();
        const response: TranscodeResult = { id: msg.id, ok: true };
        (self as any).postMessage(response);
      } catch (error) {
        const response: TranscodeResult = { 
          id: msg.id, 
          ok: false, 
          error: String(error) 
        };
        (self as any).postMessage(response);
      }
      break;
      
    case 'transcode':
      try {
        await initBasis();
        
        if (!msg.data || !msg.format) {
          throw new Error('Missing data or format');
        }
        
        const { KTX2File, TranscoderTextureFormat } = BasisModule;
        
        // Create KTX2 file object
        const ktx2 = new KTX2File(new Uint8Array(msg.data));
        
        if (!ktx2.isValid()) {
          throw new Error('Invalid KTX2 file');
        }
        
        // Get file info
        const imageCount = ktx2.getNumImages();
        const levelCount = ktx2.getNumLevels(0);
        
        // Map format to Basis enum
        const getTranscoderFormat = (): number => {
          switch (msg.format) {
            case 'astc':
              return TranscoderTextureFormat.ASTC_4x4;
            case 'bc7':
              return TranscoderTextureFormat.BC7_M5;
            case 'etc2':
              return TranscoderTextureFormat.ETC1_RGB;
            case 'rgba':
            default:
              return TranscoderTextureFormat.RGBA32;
          }
        };
        
        const transcoderFormat = getTranscoderFormat();
        const levels = [];
        
        // Transcode each mip level
        for (let level = 0; level < levelCount; level++) {
          const width = ktx2.getImageWidth(0, level);
          const height = ktx2.getImageHeight(0, level);
          
          // Get transcoded size
          const size = ktx2.getImageTranscodedSizeInBytes(0, level, transcoderFormat);
          const data = new Uint8Array(size);
          
          // Transcode the image
          const success = ktx2.transcodeImage(
            data,
            0,     // imageIndex
            level, // levelIndex
            transcoderFormat,
            0,     // getAlphaForOpaqueFormats
            0      // channel (for UASTC RG)
          );
          
          if (!success) {
            throw new Error(`Transcode failed at level ${level}`);
          }
          
          levels.push({
            width,
            height,
            data: data.buffer
          });
        }
        
        // Clean up
        ktx2.close();
        
        // Send result back with transferable objects
        const response: TranscodeResult = {
          id: msg.id,
          ok: true,
          levels
        };
        
        // @ts-ignore - Transfer ownership of ArrayBuffers
        (self as any).postMessage(response, levels.map(l => l.data));
        
      } catch (error) {
        const response: TranscodeResult = {
          id: msg.id,
          ok: false,
          error: String(error)
        };
        (self as any).postMessage(response);
      }
      break;
      
    case 'dispose':
      // Clean up resources if needed
      BasisModule = null;
      basisReady = null;
      const response: TranscodeResult = { id: msg.id, ok: true };
      (self as any).postMessage(response);
      break;
  }
};

// Log that worker is ready
console.log('[BasisWorker] Worker initialized and ready');
