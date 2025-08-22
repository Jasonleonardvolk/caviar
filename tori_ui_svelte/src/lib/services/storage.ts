// Storage adapter hub - allows switching between local FS and S3
import { env } from '$env/dynamic/private';
import type { Handle } from '@sveltejs/kit';

export interface StorageAdapter {
  save(file: File, userId: string): Promise<string>;
  list(userId: string): Promise<string[]>;
  get(path: string): Promise<Buffer | null>;
  delete(path: string): Promise<boolean>;
}

// Local filesystem implementation
class LocalStorageAdapter implements StorageAdapter {
  private uploadDir: string;
  
  constructor() {
    this.uploadDir = env.LOCAL_UPLOAD_DIR || 'var/uploads';
  }
  
  async save(file: File, userId: string): Promise<string> {
    const { writeFile } = await import('fs/promises');
    const { join } = await import('path');
    const { existsSync, mkdirSync } = await import('fs');
    
    // Ensure upload directory exists
    const userDir = join(this.uploadDir, userId);
    if (!existsSync(userDir)) {
      mkdirSync(userDir, { recursive: true });
    }
    
    // Generate unique filename
    const timestamp = Date.now();
    const filename = `${timestamp}_${file.name}`;
    const filepath = join(userDir, filename);
    
    // Save file
    const buffer = Buffer.from(await file.arrayBuffer());
    await writeFile(filepath, buffer);
    
    return filepath;
  }
  
  async list(userId: string): Promise<string[]> {
    const { readdir } = await import('fs/promises');
    const { join } = await import('path');
    const { existsSync } = await import('fs');
    
    const userDir = join(this.uploadDir, userId);
    if (!existsSync(userDir)) {
      return [];
    }
    
    return await readdir(userDir);
  }
  
  async get(path: string): Promise<Buffer | null> {
    const { readFile } = await import('fs/promises');
    const { existsSync } = await import('fs');
    
    if (!existsSync(path)) {
      return null;
    }
    
    return await readFile(path);
  }
  
  async delete(path: string): Promise<boolean> {
    const { unlink } = await import('fs/promises');
    const { existsSync } = await import('fs');
    
    if (!existsSync(path)) {
      return false;
    }
    
    await unlink(path);
    return true;
  }
}

// S3 implementation (stub for now)
class S3StorageAdapter implements StorageAdapter {
  async save(file: File, userId: string): Promise<string> {
    // TODO: Implement S3 upload
    throw new Error('S3 storage not yet implemented');
  }
  
  async list(userId: string): Promise<string[]> {
    // TODO: Implement S3 list
    throw new Error('S3 storage not yet implemented');
  }
  
  async get(path: string): Promise<Buffer | null> {
    // TODO: Implement S3 get
    throw new Error('S3 storage not yet implemented');
  }
  
  async delete(path: string): Promise<boolean> {
    // TODO: Implement S3 delete
    throw new Error('S3 storage not yet implemented');
  }
}

// Factory function to get the appropriate adapter
export function getStorageAdapter(): StorageAdapter {
  const storageType = env.STORAGE_TYPE || 'local';
  
  switch (storageType) {
    case 's3':
      return new S3StorageAdapter();
    case 'local':
    default:
      return new LocalStorageAdapter();
  }
}

// Export singleton instance
export const storage = getStorageAdapter();