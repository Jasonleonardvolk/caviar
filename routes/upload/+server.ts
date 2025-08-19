import { error, json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { writeFile, mkdir } from 'fs/promises';
import { existsSync } from 'fs';
import { join } from 'path';

// Simple PDF text extraction (for production, consider using pdf-parse or similar)
async function extractTextFromPDF(buffer: ArrayBuffer): Promise<string> {
  // For now, return a placeholder that indicates PDF processing
  // In production, integrate with pdf-parse library or similar
  const filename = 'uploaded-document.pdf';
  
  // Basic metadata extraction from PDF header
  const uint8Array = new Uint8Array(buffer);
  const pdfHeader = new TextDecoder().decode(uint8Array.slice(0, 1024));
  
  // Extract any visible text patterns (very basic)
  const textContent = new TextDecoder('utf-8', { fatal: false }).decode(uint8Array);
  const extractedText = textContent.replace(/[^\x20-\x7E\n\r]/g, ' ').trim();
  
  return extractedText.length > 100 ? extractedText.substring(0, 2000) : 
         `PDF Document: ${filename}\nContent extraction requires pdf-parse library for full text analysis.\nDocument uploaded successfully and ready for processing.`;
}

// Extract concepts from document text
function extractConcepts(text: string, filename: string): string[] {
  const concepts: string[] = [];
  const lowerText = text.toLowerCase();
  const lowerFilename = filename.toLowerCase();
  
  // File type concepts
  if (filename.endsWith('.pdf')) concepts.push('PDF Document');
  
  // Academic/research concepts
  if (lowerText.includes('abstract') || lowerText.includes('introduction') || lowerText.includes('methodology')) {
    concepts.push('Academic Paper');
  }
  if (lowerText.includes('research') || lowerText.includes('study') || lowerText.includes('analysis')) {
    concepts.push('Research');
  }
  if (lowerText.includes('conclusion') || lowerText.includes('results') || lowerText.includes('findings')) {
    concepts.push('Study Results');
  }
  
  // Technical concepts
  if (lowerText.includes('algorithm') || lowerText.includes('method') || lowerText.includes('approach')) {
    concepts.push('Technical Method');
  }
  if (lowerText.includes('ai') || lowerText.includes('artificial intelligence') || lowerText.includes('machine learning')) {
    concepts.push('Artificial Intelligence');
  }
  if (lowerText.includes('neural') || lowerText.includes('network') || lowerText.includes('deep learning')) {
    concepts.push('Neural Networks');
  }
  
  // Domain-specific concepts
  if (lowerText.includes('cognitive') || lowerText.includes('consciousness') || lowerText.includes('memory')) {
    concepts.push('Cognitive Science');
  }
  if (lowerText.includes('philosophy') || lowerText.includes('ethics') || lowerText.includes('theory')) {
    concepts.push('Philosophy');
  }
  if (lowerText.includes('data') || lowerText.includes('dataset') || lowerText.includes('statistics')) {
    concepts.push('Data Science');
  }
  
  // Filename-based concepts
  if (lowerFilename.includes('report')) concepts.push('Report');
  if (lowerFilename.includes('manual') || lowerFilename.includes('guide')) concepts.push('Documentation');
  if (lowerFilename.includes('spec') || lowerFilename.includes('requirement')) concepts.push('Specification');
  
  // Always include base document concept
  concepts.push('ScholarSphere Document');
  
  return [...new Set(concepts)]; // Remove duplicates
}

// Trigger ELFIN++ onUpload hook
function triggerELFINUpload(filename: string, text: string, concepts: string[]) {
  try {
    // Dispatch custom event for ELFIN++ processing
    if (typeof window !== 'undefined') {
      window.dispatchEvent(new CustomEvent('tori:upload', { 
        detail: { 
          filename, 
          text: text.substring(0, 1000), // Truncate for event
          concepts,
          timestamp: new Date(),
          source: 'scholarsphere'
        }
      }));
    }
    
    // Server-side ELFIN++ notification (if running)
    console.log('🧬 ELFIN++ onUpload triggered:', {
      filename,
      conceptCount: concepts.length,
      textLength: text.length
    });
    
    return true;
  } catch (err) {
    console.warn('ELFIN++ hook failed:', err);
    return false;
  }
}

export const POST: RequestHandler = async ({ request, locals }) => {
  // 🛡️ Security: Check admin role
  if (!locals.user || locals.user.role !== 'admin') {
    throw error(403, 'Admin access required for document uploads');
  }
  
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File;
    
    if (!file) {
      throw error(400, 'No file uploaded');
    }
    
    // 🛡️ Security: Validate file type and size
    const maxSize = 50 * 1024 * 1024; // 50MB limit
    if (file.size > maxSize) {
      throw error(400, 'File too large (max 50MB)');
    }
    
    const allowedTypes = ['application/pdf', 'text/plain', 'application/json'];
    if (!allowedTypes.includes(file.type) && !file.name.toLowerCase().endsWith('.pdf')) {
      throw error(400, 'Unsupported file type. PDF, TXT, and JSON files only.');
    }
    
    // 🛡️ Security: Sanitize filename
    const sanitizedFilename = file.name.replace(/[^a-zA-Z0-9.-]/g, '_').substring(0, 100);
    const timestamp = Date.now();
    const uniqueFilename = `${timestamp}_${sanitizedFilename}`;
    
    // Create storage directory
    const uploadDir = join(process.cwd(), 'data', 'sphere', 'admin');
    if (!existsSync(uploadDir)) {
      await mkdir(uploadDir, { recursive: true });
    }
    
    // Save file to disk
    const filePath = join(uploadDir, uniqueFilename);
    const arrayBuffer = await file.arrayBuffer();
    await writeFile(filePath, new Uint8Array(arrayBuffer));
    
    // Extract text content
    let extractedText: string;
    if (file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf')) {
      extractedText = await extractTextFromPDF(arrayBuffer);
    } else if (file.type === 'text/plain') {
      extractedText = new TextDecoder().decode(arrayBuffer);
    } else if (file.type === 'application/json') {
      extractedText = `JSON Document: ${file.name}\n` + new TextDecoder().decode(arrayBuffer);
    } else {
      extractedText = `Document: ${file.name}\nBinary content - ${file.size} bytes`;
    }
    
    // Extract concepts from content
    const concepts = extractConcepts(extractedText, file.name);
    
    // Trigger ELFIN++ processing
    const elfinTriggered = triggerELFINUpload(file.name, extractedText, concepts);
    
    // Prepare response data
    const documentData = {
      id: `doc_${timestamp}`,
      filename: file.name,
      uniqueFilename,
      size: file.size,
      type: file.type,
      concepts,
      extractedText: extractedText.substring(0, 500) + (extractedText.length > 500 ? '...' : ''),
      uploadedAt: new Date().toISOString(),
      uploadedBy: locals.user.name,
      filePath,
      elfinTriggered,
      summary: `${file.type} document with ${concepts.length} concepts extracted`
    };
    
    // Log successful upload
    console.log('📚 ScholarSphere Upload Complete:', {
      filename: file.name,
      size: file.size,
      concepts: concepts.length,
      elfinTriggered
    });
    
    return json({
      success: true,
      message: 'Document uploaded and processed successfully',
      document: documentData
    });
    
  } catch (err) {
    console.error('ScholarSphere upload failed:', err);
    
    if (err instanceof Error && err.message.includes('Admin access required')) {
      throw err;
    }
    
    throw error(500, 'Upload processing failed: ' + (err instanceof Error ? err.message : 'Unknown error'));
  }
};
