// Fix the chat to use enriched concepts with context instead of just names
const fs = require('fs');
const path = require('path');

console.log('🔧 Fixing Chat to Use PDF Context (Safe Version)');
console.log('================================================');

const pagePath = path.join(__dirname, 'src', 'routes', '+page.svelte');
let content = fs.readFileSync(pagePath, 'utf8');

// Check if already fixed
if (content.includes('allEnrichedConcepts')) {
    console.log('✅ Chat already fixed to use enriched concepts!');
    process.exit(0);
}

// Find the context creation section
const searchStr = "currentConcepts: [...new Set($conceptMesh.flatMap(d => d.concepts))],";
const replaceIndex = content.indexOf(searchStr);

if (replaceIndex === -1) {
    console.error('❌ Could not find currentConcepts line');
    process.exit(1);
}

// Create the replacement that maintains compatibility
const replacement = `// FIX: Get enriched concepts with context for PDFs, but also keep simple concepts
        currentConcepts: (() => {
          // Get all enriched concepts (these have context)
          const enriched = $conceptMesh
            .filter(d => d.enrichedConcepts)
            .flatMap(d => d.enrichedConcepts);
          
          // Also get simple string concepts for backward compatibility
          const simple = $conceptMesh
            .filter(d => !d.enrichedConcepts)
            .flatMap(d => d.concepts);
          
          // Combine both - enriched concepts have the context we need
          return [...enriched, ...simple];
        })(),`;

// Replace the line
content = content.substring(0, replaceIndex) + replacement + content.substring(replaceIndex + searchStr.length);

// Write the fixed file
fs.writeFileSync(pagePath, content);

console.log('✅ Fixed! The chat now uses enriched concepts with context.');
console.log('');
console.log('📝 What changed:');
console.log('   - The chat now receives BOTH enriched concepts (with context) AND simple concepts');
console.log('   - PDF concepts uploaded through ScholarSphere will have full context');
console.log('   - Other concepts will still work as before (backward compatible)');
console.log('   - The homepage will look and work exactly the same!');
console.log('');
console.log('🎯 Next steps:');
console.log('1. Restart your dev server: npm run dev');
console.log('2. Upload a PDF through ScholarSphere');
console.log('3. Ask about content from the PDF');
console.log('4. The AI should now be able to quote and use the actual PDF text!');
console.log('');
console.log('🔍 To verify it\'s working:');
console.log('1. Open browser console (F12)');
console.log('2. Look for "PDF CONTENT EXTRACTION" logs');
console.log('3. You should see "pdfConcepts" count > 0 when asking about PDFs');
console.log('4. You should see "RELEVANT PDF CONTENT FOUND" when matches occur');
