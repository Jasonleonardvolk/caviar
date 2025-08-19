/**
 * Minimal IVF demuxer for AV1 video streams
 * IVF is a simple container format used for raw video frames
 * Reference: https://wiki.multimedia.cx/index.php/IVF
 */

/**
 * IVF file header structure
 */
export interface IVFHeader {
  codec: string;        // 4-byte FourCC (e.g., 'AV01')
  width: number;        // Frame width
  height: number;       // Frame height
  timebaseNum: number;  // Timebase numerator
  timebaseDen: number;  // Timebase denominator
  frameCount: number;   // Total number of frames
  headerSize: number;   // Size of header in bytes
}

/**
 * IVF frame data
 */
export interface IVFFrame {
  timestamp: number;    // Presentation timestamp in microseconds
  data: Uint8Array;     // Frame data
  frameNumber: number;  // Frame index
}

/**
 * Parse IVF file header
 */
export function parseIVFHeader(buffer: ArrayBuffer): IVFHeader {
  if (buffer.byteLength < 32) {
    throw new Error('Buffer too small for IVF header');
  }
  
  const view = new DataView(buffer);
  
  // Check signature (DKIF)
  const signature = String.fromCharCode(
    view.getUint8(0),
    view.getUint8(1),
    view.getUint8(2),
    view.getUint8(3)
  );
  
  if (signature !== 'DKIF') {
    throw new Error(`Invalid IVF signature: ${signature}`);
  }
  
  // Read version (should be 0)
  const version = view.getUint16(4, true);
  if (version !== 0) {
    console.warn(`Unexpected IVF version: ${version}`);
  }
  
  // Read header size
  const headerSize = view.getUint16(6, true);
  
  // Read FourCC codec identifier
  const codec = String.fromCharCode(
    view.getUint8(8),
    view.getUint8(9),
    view.getUint8(10),
    view.getUint8(11)
  );
  
  // Read dimensions
  const width = view.getUint16(12, true);
  const height = view.getUint16(14, true);
  
  // Read timebase (frame rate = den/num)
  const timebaseDen = view.getUint32(16, true);
  const timebaseNum = view.getUint32(20, true);
  
  // Read frame count
  const frameCount = view.getUint32(24, true);
  
  // Unused 4 bytes at offset 28
  
  return {
    codec,
    width,
    height,
    timebaseNum,
    timebaseDen,
    frameCount,
    headerSize
  };
}

/**
 * Parse IVF frames from buffer
 * Generator function for memory efficiency
 */
export function* parseIVFFrames(
  buffer: ArrayBuffer,
  header: IVFHeader
): Generator<IVFFrame> {
  const view = new DataView(buffer);
  let offset = header.headerSize;
  let frameNumber = 0;
  
  // Calculate microseconds per frame
  const microsecondsPerFrame = header.timebaseNum && header.timebaseDen
    ? Math.round((1000000 * header.timebaseNum) / header.timebaseDen)
    : 41666; // Default to ~24fps
  
  while (offset + 12 <= buffer.byteLength) {
    // Read frame header (12 bytes)
    const frameSize = view.getUint32(offset, true);
    offset += 4;
    
    const timestampLo = view.getUint32(offset, true);
    offset += 4;
    
    const timestampHi = view.getUint32(offset, true);
    offset += 4;
    
    // Check if we have enough data for the frame
    if (offset + frameSize > buffer.byteLength) {
      console.warn(`Incomplete frame at offset ${offset}`);
      break;
    }
    
    // Extract frame data
    const frameData = new Uint8Array(buffer, offset, frameSize);
    offset += frameSize;
    
    // Calculate timestamp in microseconds
    // For simplicity, we use frame number * frame duration
    const timestamp = frameNumber * microsecondsPerFrame;
    
    yield {
      timestamp,
      data: frameData,
      frameNumber
    };
    
    frameNumber++;
  }
  
  if (frameNumber !== header.frameCount && header.frameCount > 0) {
    console.warn(`Frame count mismatch: expected ${header.frameCount}, got ${frameNumber}`);
  }
}

/**
 * Parse entire IVF file
 */
export function parseIVF(buffer: ArrayBuffer): {
  header: IVFHeader;
  frames: IVFFrame[];
} {
  const header = parseIVFHeader(buffer);
  const frames: IVFFrame[] = [];
  
  for (const frame of parseIVFFrames(buffer, header)) {
    frames.push(frame);
  }
  
  return { header, frames };
}

/**
 * IVF reader class for streaming
 */
export class IVFReader {
  private buffer: ArrayBuffer;
  private header: IVFHeader;
  private offset: number;
  private frameNumber: number;
  private microsecondsPerFrame: number;
  
  constructor(buffer: ArrayBuffer) {
    this.buffer = buffer;
    this.header = parseIVFHeader(buffer);
    this.offset = this.header.headerSize;
    this.frameNumber = 0;
    
    this.microsecondsPerFrame = this.header.timebaseNum && this.header.timebaseDen
      ? Math.round((1000000 * this.header.timebaseNum) / this.header.timebaseDen)
      : 41666;
  }
  
  get codec(): string {
    return this.header.codec;
  }
  
  get width(): number {
    return this.header.width;
  }
  
  get height(): number {
    return this.header.height;
  }
  
  get frameCount(): number {
    return this.header.frameCount;
  }
  
  get frameRate(): number {
    if (this.header.timebaseNum && this.header.timebaseDen) {
      return this.header.timebaseDen / this.header.timebaseNum;
    }
    return 24; // Default
  }
  
  /**
   * Check if there are more frames
   */
  hasNext(): boolean {
    return this.offset + 12 <= this.buffer.byteLength;
  }
  
  /**
   * Read next frame
   */
  nextFrame(): IVFFrame | null {
    if (!this.hasNext()) {
      return null;
    }
    
    const view = new DataView(this.buffer);
    
    // Read frame header
    const frameSize = view.getUint32(this.offset, true);
    this.offset += 12; // Skip frame header
    
    // Check bounds
    if (this.offset + frameSize > this.buffer.byteLength) {
      console.warn(`Incomplete frame at offset ${this.offset}`);
      return null;
    }
    
    // Extract frame data
    const frameData = new Uint8Array(this.buffer, this.offset, frameSize);
    this.offset += frameSize;
    
    const timestamp = this.frameNumber * this.microsecondsPerFrame;
    const frame: IVFFrame = {
      timestamp,
      data: frameData,
      frameNumber: this.frameNumber
    };
    
    this.frameNumber++;
    return frame;
  }
  
  /**
   * Reset to beginning
   */
  reset(): void {
    this.offset = this.header.headerSize;
    this.frameNumber = 0;
  }
  
  /**
   * Seek to specific frame
   */
  seekToFrame(targetFrame: number): boolean {
    if (targetFrame < 0 || targetFrame >= this.header.frameCount) {
      return false;
    }
    
    // Reset and skip frames
    this.reset();
    
    for (let i = 0; i < targetFrame; i++) {
      if (!this.skipFrame()) {
        return false;
      }
    }
    
    return true;
  }
  
  /**
   * Skip current frame without decoding
   */
  private skipFrame(): boolean {
    if (!this.hasNext()) {
      return false;
    }
    
    const view = new DataView(this.buffer);
    const frameSize = view.getUint32(this.offset, true);
    this.offset += 12 + frameSize;
    this.frameNumber++;
    
    return this.offset <= this.buffer.byteLength;
  }
  
  /**
   * Get all frames at once
   */
  getAllFrames(): IVFFrame[] {
    this.reset();
    const frames: IVFFrame[] = [];
    
    let frame = this.nextFrame();
    while (frame !== null) {
      frames.push(frame);
      frame = this.nextFrame();
    }
    
    return frames;
  }
}

/**
 * Load IVF file from URL
 */
export async function loadIVF(url: string): Promise<IVFReader> {
  const response = await fetch(url);
  
  if (!response.ok) {
    throw new Error(`Failed to load IVF: ${response.status} ${response.statusText}`);
  }
  
  const buffer = await response.arrayBuffer();
  return new IVFReader(buffer);
}

/**
 * Validate if buffer contains valid IVF data
 */
export function isValidIVF(buffer: ArrayBuffer): boolean {
  if (buffer.byteLength < 32) {
    return false;
  }
  
  const view = new DataView(buffer);
  const signature = String.fromCharCode(
    view.getUint8(0),
    view.getUint8(1),
    view.getUint8(2),
    view.getUint8(3)
  );
  
  return signature === 'DKIF';
}

/**
 * Get codec name from FourCC
 */
export function getCodecName(fourcc: string): string {
  const codecs: Record<string, string> = {
    'AV01': 'AV1',
    'VP80': 'VP8',
    'VP90': 'VP9',
    'H264': 'H.264/AVC',
    'HEVC': 'H.265/HEVC'
  };
  
  return codecs[fourcc] || fourcc;
}
