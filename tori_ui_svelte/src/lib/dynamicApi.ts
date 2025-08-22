// Dynamic API Port Discovery for SvelteKit
// This reads the port from the config file created by the Python API

import { readFileSync, existsSync } from 'fs';
import { join } from 'path';

interface ApiConfig {
    api_port: number;
    api_url: string;
    timestamp: number;
    status: string;
}

function getApiUrl(): string {
    const configPath = join(process.cwd(), '..', 'api_port.json');
    
    // Try to read dynamic config
    if (existsSync(configPath)) {
        try {
            const configData = readFileSync(configPath, 'utf-8');
            const config: ApiConfig = JSON.parse(configData);
            
            console.log(`🔍 [DYNAMIC] Found API config: ${config.api_url}`);
            return config.api_url;
        } catch (error) {
  const msg = error instanceof Error ? (error instanceof Error ? (error instanceof Error ? (error instanceof Error ? error.message : String(error)) : String(error)) : String(error)) : String(error);
            console.warn('⚠️ [DYNAMIC] Failed to read API config:', error);
        
}
    }
    
    // Fallback to default ports in order of preference
    const fallbackPorts = [8002, 8003, 8004, 8005];
    console.log(`🔄 [DYNAMIC] Using fallback ports: ${fallbackPorts.join(', ')}`);
    
    // Return the first fallback (SvelteKit will try multiple)
    return `http://localhost:${fallbackPorts[0]}`;
}

async function callDynamicPythonAPI(buffer: ArrayBuffer, filename: string) {
    const configPath = join(process.cwd(), '..', 'api_port.json');
    
    // Get potential URLs to try
    const urlsToTry: string[] = [];
    
    // First priority: Dynamic config
    if (existsSync(configPath)) {
        try {
            const configData = readFileSync(configPath, 'utf-8');
            const config: ApiConfig = JSON.parse(configData);
            urlsToTry.push(config.api_url);
            console.log(`🎯 [DYNAMIC] Will try configured URL: ${config.api_url}`);
        } catch (error) {
  const msg = error instanceof Error ? (error instanceof Error ? (error instanceof Error ? (error instanceof Error ? error.message : String(error)) : String(error)) : String(error)) : String(error);
            console.warn('⚠️ [DYNAMIC] Failed to read API config');
        
}
    }
    
    // Add fallback URLs
    const fallbackPorts = [8002, 8003, 8004, 8005];
    for (const port of fallbackPorts) {
        const url = `http://localhost:${port}`;
        if (!urlsToTry.includes(url)) {
            urlsToTry.push(url);
        }
    }
    
    console.log(`🔍 [DYNAMIC] Will try URLs in order: ${urlsToTry.join(', ')}`);
    
    // Try each URL until one works
    for (let i = 0; i < urlsToTry.length; i++) {
        const baseUrl = urlsToTry[i];
        const extractUrl = `${baseUrl}/extract`;
        
        try {
            console.log(`🚀 [DYNAMIC] Attempt ${i + 1}: ${extractUrl}`);
            
            const formData = new FormData();
            const blob = new Blob([buffer], { type: 'application/pdf' });
            formData.append('file', blob, filename);
            
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 30000); // 30s timeout per attempt
            
            const response = await fetch(extractUrl, {
                method: 'POST',
                body: formData,
                signal: controller.signal,
            });
            
            clearTimeout(timeoutId);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const result = await response.json();
            console.log(`✅ [DYNAMIC] SUCCESS on ${extractUrl}`);
            
            return {
                concepts: result.concepts || [],
                extractionMethod: 'enhanced_python_pipeline',
                success: true,
                processingTime: result.processing_time_seconds || 0,
                conceptCount: result.concept_count || 0,
                // Pass through all original data
                ...result,
                usedUrl: baseUrl  // Track which URL worked
            };
            
        } catch (error) {
  const msg = error instanceof Error ? (error instanceof Error ? (error instanceof Error ? (error instanceof Error ? error.message : String(error)) : String(error)) : String(error)) : String(error);
            console.log(`❌ [DYNAMIC] Failed ${extractUrl
}: ${(error instanceof Error ? (error instanceof Error ? (error instanceof Error ? error.message : String(error)) : String(error)) : String(error))}`);
            
            // If this was our last attempt, throw the error
            if (i === urlsToTry.length - 1) {
                throw new Error(`All API URLs failed. Last error: ${(error instanceof Error ? (error instanceof Error ? (error instanceof Error ? error.message : String(error)) : String(error)) : String(error))}`);
            }
            
            // Otherwise, try the next URL
            continue;
        }
    }
}

export { getApiUrl, callDynamicPythonAPI };
