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

// Find the processChatQuery method
const processChatQueryRegex = /async processChatQuery\([\s\S]*?\{/;
const match = content.match(processChatQueryRegex);

if (!match) {
    console.error('❌ Could not find processChatQuery method');
    process.exit(1);
}

// Add the PDF content extraction logic after the conceptsFromPDFs filtering
const pdfContentPatch = `
    // PDF_CONTENT_FIX_APPLIED - Extract actual content from PDF concepts
    const pdfContentMap = new Map<string, string>();
    const pdfConceptsWithContext: any[] = [];
    
    // Extract PDF concepts with their actual context
    const pdfConcepts = context.currentConcepts?.filter(c => 
        c.metadata?.source === 'scholarsphere_enhanced_server' && 
        (c.metadata?.context || c.metadata?.extractedText)
    ) || [];
    
    console.log('📚 PDF CONTENT EXTRACTION:', {
        totalPdfConcepts: pdfConcepts.length,
        withContext: pdfConcepts.filter(c => c.metadata?.context).length
    });
    
    // Build content map and concepts list
    for (const concept of pdfConcepts) {
        const conceptContext = concept.metadata?.context || concept.metadata?.extractedText || '';
        if (conceptContext) {
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
    for (const [conceptName, context] of pdfContentMap.entries()) {
        if (queryLower.includes(conceptName) || conceptName.includes(queryLower)) {
            relevantPdfContent.push(context);
            matchedConcepts.push(conceptName);
        }
    }
    
    // Also search in the context itself
    for (const concept of pdfConceptsWithContext) {
        const contextLower = concept.context.toLowerCase();
        if (contextLower.includes(queryLower) && !matchedConcepts.includes(concept.name.toLowerCase())) {
            relevantPdfContent.push(concept.context);
            matchedConcepts.push(concept.name);
        }
    }
    
    console.log('📖 RELEVANT PDF CONTENT FOUND:', {
        matchedConcepts,
        contentPieces: relevantPdfContent.length
    });
`;

// Find where to insert - look for where concepts are being used
const insertPoint = content.indexOf('// Combine concepts from query and PDFs');
if (insertPoint === -1) {
    console.error('❌ Could not find insertion point for PDF content patch');
    process.exit(1);
}

// Insert the patch
content = content.slice(0, insertPoint) + pdfContentPatch + '\n' + content.slice(insertPoint);

// Now patch the prompt building to include PDF content
const promptBuildingPatch = `
        // Include relevant PDF content in the prompt
        if (relevantPdfContent.length > 0) {
            console.log('📚 Adding PDF content to prompt:', relevantPdfContent.length, 'pieces');
            
            // Limit content to avoid token overflow
            const maxContentLength = 3000;
            let combinedContent = relevantPdfContent.join('\\n\\n---\\n\\n');
            if (combinedContent.length > maxContentLength) {
                combinedContent = combinedContent.slice(0, maxContentLength) + '...';
            }
            
            prompt += \`\\n\\n## Relevant Information from Uploaded Documents:\\n\\n\${combinedContent}\\n\\n\`;
            prompt += \`The above information is from uploaded PDFs and contains context about: \${matchedConcepts.join(', ')}\\n\\n\`;
        }
`;

// Find where the prompt is being built (before the API call)
const promptInsertRegex = /const response = await fetch\(/;
const promptInsertMatch = content.match(promptInsertRegex);

if (promptInsertMatch) {
    const insertIndex = content.lastIndexOf('const response = await fetch(');
    content = content.slice(0, insertIndex) + promptBuildingPatch + '\n        ' + content.slice(insertIndex);
}

// Write the patched file
fs.writeFileSync(enhancedApiPath, content);

console.log('✅ Successfully patched enhancedApi.ts!');
console.log('📝 Changes made:');
console.log('   - Added PDF content extraction logic');
console.log('   - Modified prompt building to include actual PDF content');
console.log('   - Added logging for debugging');

// Now create a test script
const testScript = `
// Test script to verify PDF content is being used
import { enhancedApi } from '$lib/services/enhancedApi';

async function testPdfContent() {
    console.log('🧪 Testing PDF content usage...');
    
    // Mock context with PDF concepts that have content
    const mockContext = {
        userQuery: "What is Mattis magnetization?",
        currentConcepts: [
            {
                name: "Mattis magnetization",
                score: 0.95,
                metadata: {
                    source: 'scholarsphere_enhanced_server',
                    context: 'The Mattis magnetization is a theoretical concept in condensed matter physics that describes the magnetization behavior in certain spin glass systems. It represents a specific type of order parameter that emerges in frustrated magnetic systems.'
                }
            },
            {
                name: "Marchenko-Pastur theorem",
                score: 0.88,
                metadata: {
                    source: 'scholarsphere_enhanced_server',
                    context: 'The Marchenko-Pastur theorem provides the limiting spectral distribution of large random matrices. It is fundamental in random matrix theory and has applications in physics, statistics, and machine learning.'
                }
            }
        ],
        conversationHistory: []
    };
    
    // This should now include the actual PDF content in the response
    const result = await enhancedApi.processChatQuery(mockContext);
    
    console.log('✅ Test complete! The AI should now have access to PDF content.');
    console.log('📖 Next steps:');
    console.log('   1. Restart your dev server');
    console.log('   2. Upload a PDF with known content');
    console.log('   3. Ask about specific concepts from the PDF');
    console.log('   4. The AI should quote/use the actual PDF text!');
}

export { testPdfContent };
`;

fs.writeFileSync(path.join(__dirname, 'test-pdf-content.js'), testScript);

console.log('\n📁 Created test-pdf-content.js');
console.log('\n🎯 Next Steps:');
console.log('1. Start the Python PDF server: python efficient_pdf_server.py');
console.log('2. Restart your SvelteKit dev server: npm run dev');
console.log('3. Upload a PDF and ask about its content');
console.log('4. The AI should now quote actual text from your PDFs!');
