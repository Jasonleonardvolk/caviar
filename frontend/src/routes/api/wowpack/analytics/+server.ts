import type { RequestHandler } from './$types';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';

const ROOT = path.resolve(process.cwd(), '..');
const CONTENT_DIR = path.join(ROOT, 'content', 'wowpack');

interface VideoStats {
  filename: string;
  size: number;
  sizeGB: number;
  type: string;
}

interface Analytics {
  timestamp: string;
  storage: {
    masters: { count: number; totalSizeGB: number; files: VideoStats[] };
    av1: { count: number; totalSizeGB: number; files: VideoStats[] };
    hdr10: { count: number; totalSizeGB: number; files: VideoStats[] };
    sdr: { count: number; totalSizeGB: number; files: VideoStats[] };
    output: { count: number; totalSizeGB: number; files: VideoStats[] };
  };
  pipeline: {
    ffmpegInstalled: boolean;
    encodingScriptsCount: number;
    totalVideos: number;
    totalSizeGB: number;
  };
  recommendations: string[];
}

async function analyzeDirectory(dir: string, pattern?: RegExp): Promise<VideoStats[]> {
  const files: VideoStats[] = [];
  
  if (!fs.existsSync(dir)) return files;
  
  try {
    const entries = await fsp.readdir(dir);
    for (const entry of entries) {
      if (pattern && !pattern.test(entry)) continue;
      
      const fullPath = path.join(dir, entry);
      const stats = await fsp.stat(fullPath);
      
      if (stats.isFile()) {
        files.push({
          filename: entry,
          size: stats.size,
          sizeGB: stats.size / (1024 * 1024 * 1024),
          type: path.extname(entry).replace('.', '').toUpperCase()
        });
      }
    }
  } catch (error) {
    console.error(`Error analyzing ${dir}:`, error);
  }
  
  return files;
}

export const GET: RequestHandler = async () => {
  const analytics: Analytics = {
    timestamp: new Date().toISOString(),
    storage: {
      masters: { count: 0, totalSizeGB: 0, files: [] },
      av1: { count: 0, totalSizeGB: 0, files: [] },
      hdr10: { count: 0, totalSizeGB: 0, files: [] },
      sdr: { count: 0, totalSizeGB: 0, files: [] },
      output: { count: 0, totalSizeGB: 0, files: [] }
    },
    pipeline: {
      ffmpegInstalled: fs.existsSync(path.join(ROOT, 'tools', 'ffmpeg', 'ffmpeg.exe')),
      encodingScriptsCount: 0,
      totalVideos: 0,
      totalSizeGB: 0
    },
    recommendations: []
  };

  // Analyze masters
  const mastersDir = path.join(CONTENT_DIR, 'input');
  analytics.storage.masters.files = await analyzeDirectory(mastersDir, /\.(mov|prores)$/i);
  analytics.storage.masters.count = analytics.storage.masters.files.length;
  analytics.storage.masters.totalSizeGB = analytics.storage.masters.files.reduce((sum, f) => sum + f.sizeGB, 0);

  // Analyze AV1
  const av1Dir = path.join(CONTENT_DIR, 'video', 'av1');
  analytics.storage.av1.files = await analyzeDirectory(av1Dir, /\.mp4$/i);
  analytics.storage.av1.count = analytics.storage.av1.files.length;
  analytics.storage.av1.totalSizeGB = analytics.storage.av1.files.reduce((sum, f) => sum + f.sizeGB, 0);

  // Analyze HDR10/SDR
  const hdrDir = path.join(CONTENT_DIR, 'video', 'hdr10');
  const hdrFiles = await analyzeDirectory(hdrDir, /_hdr10\.mp4$/i);
  const sdrFiles = await analyzeDirectory(hdrDir, /_sdr\.mp4$/i);
  
  analytics.storage.hdr10.files = hdrFiles;
  analytics.storage.hdr10.count = hdrFiles.length;
  analytics.storage.hdr10.totalSizeGB = hdrFiles.reduce((sum, f) => sum + f.sizeGB, 0);
  
  analytics.storage.sdr.files = sdrFiles;
  analytics.storage.sdr.count = sdrFiles.length;
  analytics.storage.sdr.totalSizeGB = sdrFiles.reduce((sum, f) => sum + f.sizeGB, 0);

  // Analyze output
  const outputDir = path.join(CONTENT_DIR, 'output');
  analytics.storage.output.files = await analyzeDirectory(outputDir);
  analytics.storage.output.count = analytics.storage.output.files.length;
  analytics.storage.output.totalSizeGB = analytics.storage.output.files.reduce((sum, f) => sum + f.sizeGB, 0);

  // Count encoding scripts
  const encodeDir = path.join(ROOT, 'tools', 'encode');
  if (fs.existsSync(encodeDir)) {
    const scripts = await fsp.readdir(encodeDir);
    analytics.pipeline.encodingScriptsCount = scripts.filter(s => s.endsWith('.ps1')).length;
  }

  // Calculate totals
  analytics.pipeline.totalVideos = 
    analytics.storage.masters.count + 
    analytics.storage.av1.count + 
    analytics.storage.hdr10.count + 
    analytics.storage.sdr.count;
    
  analytics.pipeline.totalSizeGB = 
    analytics.storage.masters.totalSizeGB + 
    analytics.storage.av1.totalSizeGB + 
    analytics.storage.hdr10.totalSizeGB + 
    analytics.storage.sdr.totalSizeGB;

  // Generate recommendations
  if (analytics.storage.masters.count === 0) {
    analytics.recommendations.push('⚠️ Add ProRes masters to content/wowpack/input/');
  }
  if (analytics.storage.av1.count === 0 && analytics.storage.masters.count > 0) {
    analytics.recommendations.push('💡 Generate AV1 encodes for next-gen codec support');
  }
  if (analytics.storage.output.count === 0 && analytics.pipeline.totalVideos > 0) {
    analytics.recommendations.push('📦 Copy encoded files to output/ for web playback');
  }
  if (!analytics.pipeline.ffmpegInstalled) {
    analytics.recommendations.push('🔧 Install FFmpeg: Run .\\tools\\encode\\Install-FFmpeg.ps1');
  }
  if (analytics.pipeline.totalVideos > 0 && analytics.recommendations.length === 0) {
    analytics.recommendations.push('✅ Pipeline fully operational!');
  }

  return new Response(JSON.stringify(analytics, null, 2), {
    headers: { 
      'content-type': 'application/json',
      'cache-control': 'no-cache'
    }
  });
};