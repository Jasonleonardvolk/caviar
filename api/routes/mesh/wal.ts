// api/routes/mesh/wal.ts
import { Request, Response } from 'express';
import * as fs from 'fs';
import * as path from 'path';

export async function walRoute(req: Request, res: Response) {
    const { userId, event } = req.body;
    
    if (!userId || !event) {
        return res.status(400).json({ error: 'Missing userId or event' });
    }
    
    try {
        // Get user-specific WAL path
        const userDir = path.join('/mesh_store', userId);
        await fs.promises.mkdir(userDir, { recursive: true });
        
        // Create daily WAL file
        const walFilename = `wal-${new Date().toISOString().split('T')[0].replace(/-/g, '')}.log`;
        const walPath = path.join(userDir, walFilename);
        
        // Append event to WAL
        const eventLine = JSON.stringify({
            timestamp: new Date().toISOString(),
            event
        }) + '\n';
        
        await fs.promises.appendFile(walPath, eventLine);
        
        res.json({ success: true });
    } catch (error) {
        console.error('WAL error:', error);
        res.status(500).json({ error: 'Failed to write to WAL' });
    }
}
