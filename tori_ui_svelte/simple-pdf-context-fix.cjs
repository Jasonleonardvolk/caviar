// Simple fix to ensure PDF context is properly stored and used
const fs = require('fs');
const path = require('path');

console.log('🔧 Simple PDF Context Fix');
console.log('========================');

// Fix 1: Patch ScholarSpherePanel to properly store context
const scholarSpherePath = path.join(__dirname, 'src', 'lib', 'components', 'ScholarSpherePanel.svelte');
let scholarContent = fs.readFileSync(scholarSpherePath, 'utf8');

// Check if already patched
if (scholarContent.includes('// CONTEXT_FIX_APPLIED')) {
    console.log('✅ ScholarSphere already patched');
} else {
    // Find where concepts are processed
    const processConceptsMarker = 'conceptsArray = result.document.concepts.map((concept: any, index: number) => {';
    const markerIndex = scholarContent.indexOf(processConceptsMarker);
    
    if (markerIndex !== -1) {
        // Find the return statement inside the map
        const returnIndex = scholarContent.indexOf('return {', markerIndex);
        
        if (returnIndex !== -1) {
            // Add context to the returned object
            const contextPatch = `
                  // CONTEXT_FIX_APPLIED - Ensure context is preserved
                  actualContext: concept.context || concept.metadata?.context || '',`;
            
            // Insert after the return {
            scholarContent = scholarContent.slice(0, returnIndex + 8) + contextPatch + scholarContent.slice(returnIndex + 8);
            
            console.log('✅ Patched ScholarSphere concept processing');
        }
    }
    
    // Now fix the metadata passed to addConceptDiff
    const metadataMarker = 'extractedText: result.document.extractedText,';
    const metadataIndex = scholarContent.indexOf(metadataMarker);
    
    if (metadataIndex !== -1) {
        const metadataPatch = `
            conceptsWithContext: conceptsArray, // Pass full concepts with context`;
        
        scholarContent = scholarContent.slice(0, metadataIndex + metadataMarker.length) + metadataPatch + scholarContent.slice(metadataIndex + metadataMarker.length);
        console.log('✅ Patched metadata to include concepts with context');
    }
    
    fs.writeFileSync(scholarSpherePath, scholarContent);
}

// Fix 2: Update conceptMesh to handle context properly
const conceptMeshPath = path.join(__dirname, 'src', 'lib', 'stores', 'conceptMesh.ts');
let meshContent = fs.readFileSync(conceptMeshPath, 'utf8');

if (meshContent.includes('// CONTEXT_STORAGE_FIX')) {
    console.log('✅ ConceptMesh already patched');
} else {
    // Find the addConceptDiff function
    const addConceptMarker = 'export function addConceptDiff(docId: string, concepts: any[], metadata: any = {}) {';
    const funcIndex = meshContent.indexOf(addConceptMarker);
    
    if (funcIndex !== -1) {
        // Add context preservation logic
        const contextFix = `
  // CONTEXT_STORAGE_FIX - Preserve concept context
  const enrichedConcepts = concepts.map(concept => {
    if (typeof concept === 'string') {
      return concept;
    }
    
    // Ensure context is stored in a findable location
    const enriched = {
      ...concept,
      actualContext: concept.actualContext || concept.context || concept.metadata?.context || '',
      metadata: {
        ...concept.metadata,
        context: concept.actualContext || concept.context || concept.metadata?.context || '',
        extractedText: metadata.extractedText || '',
        conceptsWithContext: metadata.conceptsWithContext
      }
    };
    
    return enriched;
  });
  
  // Use enriched concepts instead of original
  concepts = enrichedConcepts;
`;
        
        // Find the opening brace and insert after it
        const braceIndex = meshContent.indexOf('{', funcIndex);
        meshContent = meshContent.slice(0, braceIndex + 1) + contextFix + meshContent.slice(braceIndex + 1);
        
        console.log('✅ Patched conceptMesh to preserve context');
    }
    
    fs.writeFileSync(conceptMeshPath, meshContent);
}

// Fix 3: Create a debug helper
const debugHelper = `
// Debug helper to check if PDF content is available
export function debugPdfContent() {
    const meshStore = get(conceptMesh);
    console.log('=== PDF CONTENT DEBUG ===');
    
    let pdfConceptCount = 0;
    let withContextCount = 0;
    
    meshStore.forEach(doc => {
        if (doc.metadata?.source === 'scholarsphere_enhanced_server') {
            console.log('📄 Document:', doc.title);
            
            if (doc.concepts && Array.isArray(doc.concepts)) {
                doc.concepts.forEach(concept => {
                    pdfConceptCount++;
                    const context = concept.actualContext || concept.context || concept.metadata?.context;
                    
                    if (context) {
                        withContextCount++;
                        console.log('  ✅', concept.name || concept, '- HAS CONTEXT:', context.substring(0, 100) + '...');
                    } else {
                        console.log('  ❌', concept.name || concept, '- NO CONTEXT');
                    }
                });
            }
        }
    });
    
    console.log('Summary:', withContextCount, 'of', pdfConceptCount, 'PDF concepts have context');
    console.log('========================');
}

// Add to window for easy access
if (typeof window !== 'undefined') {
    (window as any).debugPdfContent = debugPdfContent;
}
`;

// Append debug helper to conceptMesh
if (!meshContent.includes('debugPdfContent')) {
    fs.appendFileSync(conceptMeshPath, '\n\n' + debugHelper);
    console.log('✅ Added debug helper to conceptMesh');
}

console.log('\n✅ All fixes applied!');
console.log('\n📋 Next steps:');
console.log('1. Restart your dev server: npm run dev');
console.log('2. Open http://localhost:5173/test-pdf-upload.html');
console.log('3. Upload a PDF and check if context is shown');
console.log('4. In browser console, run: window.debugPdfContent()');
console.log('5. This will show you exactly what context is stored for each concept');
