// Fix script to patch enhancedApi.ts to use PDF content, not just concept names
const fs = require('fs');
const path = require('path');

console.log('🔧 PDF Content Usage Fix Script');
console.log('================================');

const enhancedApiPath = path.join(__dirname, 'src', 'lib', 'services', 'enhancedApi.ts');

// Read the current file
let content = fs.readFileSync(enhancedApiPath, 'utf8');

// Check if already patched
if (content.includes('// PDF_CONTENT_FIX_APPLIED')) {
    console.log('✅ Fix already applied!');
    process.exit(0);
}

console.log('📝 Patching enhancedApi.ts to use PDF content...');

// Find the generateResponse method
const generateResponseRegex = /async generateResponse\(context: ConversationContext\): Promise<EnhancedResponse> \{/;
const match = content.match(generateResponseRegex);

if (!match) {
    console.error('❌ Could not find generateResponse method');
    process.exit(1);
}

// Find the correct insertion point - after the console.log statement
const insertionMarker = 'console.log(`🧠 FULL SYSTEM: Processing query with ALL cognitive systems: "${context.userQuery}"`);';
const insertionIndex = content.indexOf(insertionMarker);

if (insertionIndex === -1) {
    console.error('❌ Could not find insertion point for PDF content patch');
    process.exit(1);
}

// Calculate where to insert (after the console.log line)
const afterMarkerIndex = insertionIndex + insertionMarker.length;

// Add the PDF content extraction logic
const pdfContentPatch = `
    
    // PDF_CONTENT_FIX_APPLIED - Extract actual content from PDF concepts
    const pdfContentMap = new Map<string, string>();
    const pdfConceptsWithContext: any[] = [];
    
    // Extract PDF concepts with their actual context - handle both array and object formats
    const currentConcepts = Array.isArray(context.currentConcepts) 
        ? context.currentConcepts 
        : (context as any).currentConcepts || [];
        
    const pdfConcepts = currentConcepts.filter((c: any) => {
        // Handle both string and object concept formats
        if (typeof c === 'string') return false;
        return c?.metadata?.source === 'scholarsphere_enhanced_server' && 
               (c?.metadata?.context || c?.metadata?.extractedText);
    });
    
    console.log('📚 PDF CONTENT EXTRACTION:', {
        totalConcepts: currentConcepts.length,
        pdfConcepts: pdfConcepts.length,
        withContext: pdfConcepts.filter((c: any) => c?.metadata?.context).length
    });
    
    // Build content map and concepts list
    for (const concept of pdfConcepts) {
        const conceptContext = concept.metadata?.context || concept.metadata?.extractedText || '';
        if (conceptContext && concept.name) {
            pdfContentMap.set(concept.name.toLowerCase(), conceptContext);
            pdfConceptsWithContext.push({
                name: concept.name,
                score: concept.score || 0.8,
                context: conceptContext,
                metadata: concept.metadata
            });
        }
    }
    
    // Find relevant PDF content based on query
    const queryLower = context.userQuery.toLowerCase();
    const relevantPdfContent: string[] = [];
    const matchedConcepts: string[] = [];
    
    // Search for query terms in PDF concepts
    for (const [conceptName, contextText] of pdfContentMap.entries()) {
        if (queryLower.includes(conceptName) || conceptName.includes(queryLower)) {
            relevantPdfContent.push(contextText);
            matchedConcepts.push(conceptName);
        }
    }
    
    // Also search in the context itself for broader matches
    for (const concept of pdfConceptsWithContext) {
        const contextLower = concept.context.toLowerCase();
        const words = queryLower.split(/\\s+/);
        const hasRelevantContent = words.some(word => 
            word.length > 3 && contextLower.includes(word)
        );
        
        if (hasRelevantContent && !matchedConcepts.includes(concept.name.toLowerCase())) {
            relevantPdfContent.push(concept.context);
            matchedConcepts.push(concept.name);
        }
    }
    
    console.log('📖 RELEVANT PDF CONTENT FOUND:', {
        matchedConcepts,
        contentPieces: relevantPdfContent.length
    });
    
    // Store PDF content in context for use in processing methods
    (context as any).pdfContent = relevantPdfContent;
    (context as any).pdfMatchedConcepts = matchedConcepts;
`;

// Insert the PDF content patch
content = content.slice(0, afterMarkerIndex) + pdfContentPatch + content.slice(afterMarkerIndex);

// Now we need to patch the simple processing method to include PDF content
// Find the enhancedSimpleProcessing method
const simpleProcessingRegex = /private async enhancedSimpleProcessing\(/;
const simpleProcessingIndex = content.indexOf('private async enhancedSimpleProcessing(');

if (simpleProcessingIndex !== -1) {
    // Find where concepts are used in this method
    const conceptsUsageMarker = '📚 USING CONCEPTS:';
    const conceptsUsageIndex = content.indexOf(conceptsUsageMarker, simpleProcessingIndex);
    
    if (conceptsUsageIndex !== -1) {
        // Find the next fetch call after this point
        const fetchIndex = content.indexOf('const response = await fetch(', conceptsUsageIndex);
        
        if (fetchIndex !== -1) {
            // Find the prompt building section (should be just before fetch)
            const promptBuildingPatch = `
        // Include relevant PDF content in the prompt
        const pdfContent = (context as any).pdfContent || [];
        const pdfMatchedConcepts = (context as any).pdfMatchedConcepts || [];
        
        if (pdfContent.length > 0) {
            console.log('📚 Adding PDF content to prompt:', pdfContent.length, 'pieces');
            
            // Limit content to avoid token overflow
            const maxContentLength = 3000;
            let combinedContent = pdfContent.join('\\n\\n---\\n\\n');
            if (combinedContent.length > maxContentLength) {
                combinedContent = combinedContent.slice(0, maxContentLength) + '...';
            }
            
            prompt += \`\\n\\n## Relevant Information from Uploaded Documents:\\n\\n\${combinedContent}\\n\\n\`;
            prompt += \`The above information is from uploaded PDFs and contains context about: \${pdfMatchedConcepts.join(', ')}\\n\\n\`;
        }
        
        `;
            
            // Insert before the fetch call
            content = content.slice(0, fetchIndex) + promptBuildingPatch + content.slice(fetchIndex);
        }
    }
}

// Write the patched file
fs.writeFileSync(enhancedApiPath, content);

console.log('✅ Successfully patched enhancedApi.ts!');
console.log('📝 Changes made:');
console.log('   - Added PDF content extraction logic in generateResponse');
console.log('   - Modified enhancedSimpleProcessing to include actual PDF content');
console.log('   - Added logging for debugging');
console.log('');
console.log('🎯 Next Steps:');
console.log('1. Start the Python PDF server: python efficient_pdf_server.py');
console.log('2. Restart your SvelteKit dev server: npm run dev');
console.log('3. Upload a PDF and ask about its content');
console.log('4. The AI should now quote actual text from your PDFs!');
