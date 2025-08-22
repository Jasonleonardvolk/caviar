import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { readFile } from 'fs/promises';
import path from 'path';

// Read the dynamic port from the config file
async function getDynamicApiUrl() {
    try {
        const configPath = path.join(process.cwd(), '..', 'api_port.json');
        const config = JSON.parse(await readFile(configPath, 'utf-8'));
        return config.api_url || 'http://localhost:8002';
    } catch {
        return 'http://localhost:8002';
    }
}

export const GET: RequestHandler = async () => {
    try {
        const apiUrl = await getDynamicApiUrl();
        console.log('🔍 Fetching extraction results from:', `${apiUrl}/list`);
        
        const response = await fetch(`${apiUrl}/list`);
        
        if (!response.ok) {
            console.error('❌ Backend /list endpoint error:', response.status);
            return json({ 
                error: 'Failed to fetch extraction results',
                status: response.status 
            }, { status: response.status });
        }
        
        const data = await response.json();
        console.log('✅ Retrieved extraction results:', data?.length || 'unknown count');
        
        return json(data);
        
    } catch (error) {
        console.error('❌ Failed to fetch extraction results:', error);
        return json({ 
            error: 'Backend extraction API not available',
            hint: 'Make sure the dynamic API server is running'
        }, { status: 503 });
    }
};