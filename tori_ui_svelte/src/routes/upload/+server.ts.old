import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { readFile } from 'fs/promises';
import { writeFile } from 'fs/promises';
import { unlink } from 'fs/promises';
import path from 'path';
import os from 'os';

// Read the dynamic port from the config file
async function getDynamicApiUrl() {
    try {
        const configPath = path.join(process.cwd(), '..', 'api_port.json');
        const config = JSON.parse(await readFile(configPath, 'utf-8'));
        return config.api_url || 'http://localhost:8002';
    } catch {
        // Fallback to default if config doesn't exist
        return 'http://localhost:8002';
    }
}

export const POST: RequestHandler = async ({ request, locals }) => {
    let tempFilePath: string | null = null;
    
    try {
        const formData = await request.formData();
        const file = formData.get('file') as File;
        
        if (!file) {
            return json({ error: 'No file provided' }, { status: 400 });
        }
        
        console.log('📤 Processing file for extraction:', file.name, file.size);
        
        // Get the current API URL
        const apiUrl = await getDynamicApiUrl();
        console.log('🔗 Using API URL:', apiUrl);
        
        // Save file temporarily
        const buffer = Buffer.from(await file.arrayBuffer());
        const tempDir = os.tmpdir();
        const timestamp = Date.now();
        const safeFilename = file.name.replace(/[^a-zA-Z0-9.-]/g, '_');
        tempFilePath = path.join(tempDir, `upload_${timestamp}_${safeFilename}`);
        
        await writeFile(tempFilePath, buffer);
        console.log('💾 Saved temporary file:', tempFilePath);
        
        // Generate a unique progress ID for WebSocket tracking
        const progressId = `progress_${timestamp}_${Math.random().toString(36).substr(2, 9)}`;
        
        // Call the extraction endpoint on the dynamic API
        const extractRequest = {
            file_path: tempFilePath,
            filename: file.name,
            content_type: file.type || 'application/pdf',
            progress_id: progressId
        };
        
        console.log('🚀 Calling extraction API:', extractRequest);
        
        const response = await fetch(`${apiUrl}/extract`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(extractRequest)
        });
        
        if (!response.ok) {
            const errorText = await response.text();
            console.error('❌ Extraction API error:', response.status, errorText);
            
            // Try to parse error as JSON
            let errorDetails;
            try {
                errorDetails = JSON.parse(errorText);
            } catch {
                errorDetails = { error: errorText };
            }
            
            return json({ 
                error: `Extraction failed: ${response.statusText}`,
                details: errorDetails,
                success: false
            }, { status: response.status });
        }
        
        const result = await response.json();
        console.log('✅ Extraction API response:', {
            success: result.success,
            conceptCount: result.concept_count,
            method: result.extraction_method
        });
        
        if (result.success || (result.concept_count && result.concept_count > 0)) {
            // Transform the extraction result to match ScholarSphere format
            const concepts = result.concept_names?.map((name: string, index: number) => ({
                name: name,
                score: 0.9 - (index * 0.02), // Decreasing scores
                method: result.extraction_method || 'purity_based_universal_pipeline',
                source: {
                    database_matched: false,
                    server_extraction: true,
                    enhanced_pipeline: true,
                    purity_analysis: true
                },
                context: `Extracted via ${result.extraction_method || 'enhanced pipeline'}`,
                metadata: {
                    extraction_method: result.extraction_method,
                    processing_time: result.processing_time_seconds,
                    server_side: true,
                    enhanced_pipeline: true,
                    purity_score: result.purity_score || 0.8
                }
            })) || [];
            
            return json({
                success: true,
                document: {
                    id: `doc_${timestamp}_${Math.random().toString(36).substr(2, 9)}`,
                    filename: file.name,
                    size: file.size,
                    uploadedAt: new Date().toISOString(),
                    uploadedBy: locals.user?.name || 'anonymous',
                    concepts: concepts,
                    conceptCount: result.concept_count || concepts.length,
                    extractedText: result.extracted_text_preview || '',
                    extractionMethod: result.extraction_method || 'purity_based_universal_pipeline',
                    enhancedExtraction: true,
                    processingTime: result.processing_time_seconds || 0,
                    summary: result.summary || `Extracted ${result.concept_count || 0} pure concepts from ${file.name}`,
                    // Additional metadata
                    conceptNames: result.concept_names || [],
                    message: `Successfully extracted ${result.concept_count || 0} concepts`,
                    progressId: progressId,
                    stats: {
                        totalConcepts: result.concept_count,
                        uniqueConcepts: result.unique_concept_count,
                        domains: result.domains_detected,
                        confidence: result.confidence_score
                    }
                }
            });
        } else {
            return json({ 
                error: result.error || 'Extraction failed - no concepts found',
                success: false,
                details: result
            }, { status: 400 });
        }
        
    } catch (error) {
        console.error('❌ Upload handler error:', error);
        
        // Check if extraction API is running
        if (error instanceof Error && error.message.includes('ECONNREFUSED')) {
            return json({ 
                error: 'Extraction API is not running. Please start the dynamic API server.',
                details: 'Run: python start_dynamic_api.py',
                success: false
            }, { status: 503 });
        }
        
        return json({ 
            error: 'Failed to process upload',
            details: error instanceof Error ? error.message : 'Unknown error',
            success: false
        }, { status: 500 });
    } finally {
        // Clean up temporary file
        if (tempFilePath) {
            try {
                await unlink(tempFilePath);
                console.log('🧹 Cleaned up temporary file:', tempFilePath);
            } catch (error) {
                console.warn('Failed to clean up temp file:', error);
            }
        }
    }
};

// Check extraction API status
export const GET: RequestHandler = async () => {
    try {
        const apiUrl = await getDynamicApiUrl();
        const response = await fetch(`${apiUrl}/health`);
        
        if (response.ok) {
            const data = await response.json();
            return json({
                status: 'operational',
                serverUrl: apiUrl,
                message: 'Extraction API is running',
                features: data.features,
                soliton_enabled: data.soliton_enabled
            });
        }
        return json({ 
            status: 'error', 
            message: 'Extraction API not responding',
            serverUrl: apiUrl
        }, { status: 503 });
    } catch (error) {
        return json({ 
            status: 'offline',
            message: 'Extraction API is not running',
            hint: 'Run: python start_dynamic_api.py'
        }, { status: 503 });
    }
};
