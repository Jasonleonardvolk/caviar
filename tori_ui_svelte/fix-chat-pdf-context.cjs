// Fix the chat to use enriched concepts with context instead of just names
const fs = require('fs');
const path = require('path');

console.log('🔧 Fixing Chat to Use PDF Context');
console.log('==================================');

const pagePath = path.join(__dirname, 'src', 'routes', '+page.svelte');
let content = fs.readFileSync(pagePath, 'utf8');

// Check if already fixed
if (content.includes('allEnrichedConcepts')) {
    console.log('✅ Chat already fixed to use enriched concepts!');
    process.exit(0);
}

// Find the line where currentConcepts is set
const oldLine = "currentConcepts: [...new Set($conceptMesh.flatMap(d => d.concepts))],";
const newLines = `// FIX: Get enriched concepts with context, not just names
      const allEnrichedConcepts = $conceptMesh
        .filter(d => d.enrichedConcepts) // Only diffs that have enriched concepts
        .flatMap(d => d.enrichedConcepts);
      
      const context = {
        userQuery: currentMessage,
        currentConcepts: allEnrichedConcepts, // Now includes full context!`;

// Find and replace
const searchStr = "const context = {";
const contextIndex = content.indexOf(searchStr);

if (contextIndex === -1) {
    console.error('❌ Could not find context creation in +page.svelte');
    process.exit(1);
}

// Find the old currentConcepts line
const oldLineIndex = content.indexOf(oldLine);
if (oldLineIndex === -1) {
    console.error('❌ Could not find currentConcepts line');
    process.exit(1);
}

// Replace from "const context = {" to just after the currentConcepts line
const beforeContext = content.substring(0, contextIndex - 6); // Include indentation
const afterCurrentConcepts = content.substring(oldLineIndex + oldLine.length);

const newContent = beforeContext + "      " + newLines + afterCurrentConcepts;

// Write the fixed file
fs.writeFileSync(pagePath, newContent);

console.log('✅ Fixed! The chat now uses enriched concepts with context.');
console.log('');
console.log('📝 What changed:');
console.log('   - Instead of just concept names, the chat now receives full concept objects');
console.log('   - Each concept now includes its "context" field from PDF extraction');
console.log('   - The enhancedApi can now access and use the actual PDF content!');
console.log('');
console.log('🎯 Next steps:');
console.log('1. Restart your dev server: npm run dev');
console.log('2. Upload a PDF through ScholarSphere');
console.log('3. Ask about content from the PDF');
console.log('4. The AI should now be able to quote and use the actual PDF text!');
console.log('');
console.log('🔍 To verify it\'s working, check the browser console for:');
console.log('   - "PDF CONTENT EXTRACTION" logs showing concepts with context');
console.log('   - "RELEVANT PDF CONTENT FOUND" when you ask about PDF topics');
