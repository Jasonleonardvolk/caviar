import { json } from '@sveltejs/kit';
import { env } from '$env/dynamic/private';
import * as fs from 'node:fs/promises';
import * as path from 'node:path';

const LOCAL_UPLOAD_DIR = env.LOCAL_UPLOAD_DIR || 'var/uploads';

export const GET = async ({ url, locals }) => {
  try {
    // Ensure directory exists
    await fs.mkdir(LOCAL_UPLOAD_DIR, { recursive: true });
    
    // Get user ID from session or use 'public' for unauthenticated
    const userId = locals?.user?.id || 'public';
    const userDir = path.join(LOCAL_UPLOAD_DIR, userId);
    
    // Check if user directory exists
    try {
      await fs.access(userDir);
    } catch {
      // Directory doesn't exist, return empty list
      return json({ 
        ok: true, 
        files: [],
        user: userId 
      });
    }
    
    // Read directory contents
    const entries = await fs.readdir(userDir, { withFileTypes: true });
    const files = await Promise.all(
      entries
        .filter(e => e.isFile())
        .map(async (e) => {
          const fullPath = path.join(userDir, e.name);
          const stats = await fs.stat(fullPath);
          return {
            name: e.name,
            size: stats.size,
            modified: stats.mtime.toISOString(),
            path: path.relative(LOCAL_UPLOAD_DIR, fullPath)
          };
        })
    );
    
    // Sort by modified date (newest first)
    files.sort((a, b) => b.modified.localeCompare(a.modified));
    
    return json({ 
      ok: true, 
      files,
      user: userId,
      count: files.length
    });
  } catch (error) {
    console.error('Error listing files:', error);
    return json({ 
      ok: false, 
      error: 'Failed to list files',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
};