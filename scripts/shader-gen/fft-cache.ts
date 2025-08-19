/**
 * FFT Table Cache Manager
 * Caches precomputed FFT tables to disk for faster startup
 */

import fs from 'fs/promises';
import path from 'path';
import os from 'os';
import { createHash } from 'crypto';

interface CachedFFTData {
    size: number;
    version: string;
    timestamp: number;
    data: {
        twiddlesForward: Float32Array;
        twiddlesInverse: Float32Array;
        bitReversal: Uint32Array;
        twiddleOffsets: Uint32Array;
    };
}

export class FFTCacheManager {
    private cacheDir: string;
    private version = '1.0.0'; // Bump when algorithm changes
    
    constructor() {
        // Windows-friendly cache location
        const appData = process.env.LOCALAPPDATA || 
                       process.env.APPDATA || 
                       path.join(os.homedir(), 'AppData', 'Local');
        
        this.cacheDir = path.join(appData, 'Tori', 'cache', 'fft');
    }
    
    async initialize(): Promise<void> {
        await fs.mkdir(this.cacheDir, { recursive: true });
    }
    
    /**
     * Get cached FFT data or generate and cache it
     */
    async getFFTData(size: number): Promise<CachedFFTData['data']> {
        const cacheKey = this.getCacheKey(size);
        const cachePath = path.join(this.cacheDir, `fft_${cacheKey}.bin`);
        
        // Try to load from cache
        try {
            const cached = await this.loadFromCache(cachePath);
            if (cached && cached.version === this.version) {
                console.log(`Loaded FFT data for size ${size} from cache`);
                return cached.data;
            }
        } catch (error) {
            // Cache miss or corrupt
        }
        
        // Generate new data
        console.log(`Generating FFT data for size ${size}...`);
        const data = this.generateFFTData(size);
        
        // Save to cache
        await this.saveToCache(cachePath, size, data);
        
        return data;
    }
    
    /**
     * Pre-generate common sizes
     */
    async prebakeCommonSizes(): Promise<void> {
        const sizes = [256, 512, 1024, 2048, 4096, 8192];
        
        console.log('Pre-baking FFT tables...');
        
        for (const size of sizes) {
            await this.getFFTData(size);
        }
        
        console.log('FFT table pre-baking complete!');
    }
    
    private getCacheKey(size: number): string {
        return `${size}_${this.version}`;
    }
    
    private async loadFromCache(cachePath: string): Promise<CachedFFTData | null> {
        try {
            const buffer = await fs.readFile(cachePath);
            
            // Parse binary format
            const view = new DataView(buffer.buffer);
            let offset = 0;
            
            // Read header
            const size = view.getUint32(offset, true); offset += 4;
            const versionLength = view.getUint32(offset, true); offset += 4;
            
            const versionBytes = new Uint8Array(buffer.buffer, offset, versionLength);
            const version = new TextDecoder().decode(versionBytes);
            offset += versionLength;
            
            const timestamp = Number(view.getBigUint64(offset, true)); offset += 8;
            
            // Read data sizes
            const twiddleSize = view.getUint32(offset, true); offset += 4;
            const bitRevSize = view.getUint32(offset, true); offset += 4;
            const offsetSize = view.getUint32(offset, true); offset += 4;
            
            // Read data
            const twiddlesForward = new Float32Array(
                buffer.buffer.slice(offset, offset + twiddleSize * 4)
            );
            offset += twiddleSize * 4;
            
            const twiddlesInverse = new Float32Array(
                buffer.buffer.slice(offset, offset + twiddleSize * 4)
            );
            offset += twiddleSize * 4;
            
            const bitReversal = new Uint32Array(
                buffer.buffer.slice(offset, offset + bitRevSize * 4)
            );
            offset += bitRevSize * 4;
            
            const twiddleOffsets = new Uint32Array(
                buffer.buffer.slice(offset, offset + offsetSize * 4)
            );
            
            return {
                size,
                version,
                timestamp,
                data: {
                    twiddlesForward,
                    twiddlesInverse,
                    bitReversal,
                    twiddleOffsets
                }
            };
            
        } catch (error) {
            return null;
        }
    }
    
    private async saveToCache(
        cachePath: string, 
        size: number, 
        data: CachedFFTData['data']
    ): Promise<void> {
        // Calculate total size
        const versionBytes = new TextEncoder().encode(this.version);
        const headerSize = 4 + 4 + versionBytes.length + 8 + 4 + 4 + 4;
        const dataSize = 
            data.twiddlesForward.byteLength +
            data.twiddlesInverse.byteLength +
            data.bitReversal.byteLength +
            data.twiddleOffsets.byteLength;
        
        const totalSize = headerSize + dataSize;
        const buffer = new ArrayBuffer(totalSize);
        const view = new DataView(buffer);
        let offset = 0;
        
        // Write header
        view.setUint32(offset, size, true); offset += 4;
        view.setUint32(offset, versionBytes.length, true); offset += 4;
        new Uint8Array(buffer, offset, versionBytes.length).set(versionBytes);
        offset += versionBytes.length;
        view.setBigUint64(offset, BigInt(Date.now()), true); offset += 8;
        
        // Write data sizes
        view.setUint32(offset, data.twiddlesForward.length, true); offset += 4;
        view.setUint32(offset, data.bitReversal.length, true); offset += 4;
        view.setUint32(offset, data.twiddleOffsets.length, true); offset += 4;
        
        // Write data
        new Float32Array(buffer, offset).set(data.twiddlesForward);
        offset += data.twiddlesForward.byteLength;
        
        new Float32Array(buffer, offset).set(data.twiddlesInverse);
        offset += data.twiddlesInverse.byteLength;
        
        new Uint32Array(buffer, offset).set(data.bitReversal);
        offset += data.bitReversal.byteLength;
        
        new Uint32Array(buffer, offset).set(data.twiddleOffsets);
        
        // Write to file
        await fs.writeFile(cachePath, new Uint8Array(buffer));
    }
    
    private generateFFTData(size: number): CachedFFTData['data'] {
        const stages = Math.log2(size);
        
        // Generate twiddle factors
        const twiddlesForward = this.generateTwiddles(size, 'forward');
        const twiddlesInverse = this.generateTwiddles(size, 'inverse');
        
        // Generate bit reversal
        const bitReversal = new Uint32Array(size);
        for (let i = 0; i < size; i++) {
            let reversed = 0;
            let temp = i;
            
            for (let j = 0; j < stages; j++) {
                reversed = (reversed << 1) | (temp & 1);
                temp >>= 1;
            }
            
            bitReversal[i] = reversed;
        }
        
        // Generate twiddle offsets
        const twiddleOffsets = new Uint32Array(stages);
        let offset = 0;
        
        for (let stage = 0; stage < stages; stage++) {
            twiddleOffsets[stage] = offset;
            const stageSize = 1 << (stage + 1);
            offset += stageSize >> 1;
        }
        
        return {
            twiddlesForward,
            twiddlesInverse,
            bitReversal,
            twiddleOffsets
        };
    }
    
    private generateTwiddles(size: number, direction: 'forward' | 'inverse'): Float32Array {
        const sign = direction === 'forward' ? -1 : 1;
        const stages = Math.log2(size);
        const data: number[] = [];
        
        for (let stage = 0; stage < stages; stage++) {
            const stageSize = 1 << (stage + 1);
            const halfStageSize = stageSize >> 1;
            
            for (let k = 0; k < halfStageSize; k++) {
                const angle = sign * 2 * Math.PI * k / stageSize;
                data.push(Math.cos(angle), Math.sin(angle));
            }
        }
        
        return new Float32Array(data);
    }
    
    /**
     * Clear cache
     */
    async clearCache(): Promise<void> {
        try {
            const files = await fs.readdir(this.cacheDir);
            
            for (const file of files) {
                if (file.startsWith('fft_') && file.endsWith('.bin')) {
                    await fs.unlink(path.join(this.cacheDir, file));
                }
            }
            
            console.log('FFT cache cleared');
        } catch (error) {
            console.error('Failed to clear cache:', error);
        }
    }
    
    /**
     * Get cache statistics
     */
    async getCacheStats(): Promise<{
        totalSize: number;
        fileCount: number;
        sizes: number[];
    }> {
        try {
            const files = await fs.readdir(this.cacheDir);
            let totalSize = 0;
            const sizes: number[] = [];
            
            for (const file of files) {
                if (file.startsWith('fft_') && file.endsWith('.bin')) {
                    const stats = await fs.stat(path.join(this.cacheDir, file));
                    totalSize += stats.size;
                    
                    // Extract size from filename
                    const match = file.match(/fft_(\d+)_/);
                    if (match) {
                        sizes.push(parseInt(match[1]));
                    }
                }
            }
            
            return {
                totalSize,
                fileCount: sizes.length,
                sizes: sizes.sort((a, b) => a - b)
            };
            
        } catch (error) {
            return {
                totalSize: 0,
                fileCount: 0,
                sizes: []
            };
        }
    }
}

// Export singleton
export const fftCache = new FFTCacheManager();

// CLI interface
if (require.main === module) {
    async function main() {
        const command = process.argv[2];
        
        await fftCache.initialize();
        
        switch (command) {
            case 'prebake':
                await fftCache.prebakeCommonSizes();
                break;
                
            case 'clear':
                await fftCache.clearCache();
                break;
                
            case 'stats':
                const stats = await fftCache.getCacheStats();
                console.log('FFT Cache Statistics:');
                console.log(`  Total size: ${(stats.totalSize / 1024 / 1024).toFixed(2)} MB`);
                console.log(`  File count: ${stats.fileCount}`);
                console.log(`  Cached sizes: ${stats.sizes.join(', ')}`);
                break;
                
            default:
                console.log('Usage: ts-node fft-cache.ts [prebake|clear|stats]');
        }
    }
    
    main().catch(console.error);
}
