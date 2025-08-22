// Restore original ScholarSphere functionality
const fs = require('fs');
const path = require('path');

console.log('🔧 Restoring Original ScholarSphere');
console.log('===================================');

const scholarSpherePath = path.join(__dirname, 'src', 'lib', 'components', 'ScholarSpherePanel.svelte');
let content = fs.readFileSync(scholarSpherePath, 'utf8');

// Remove the context fix line
if (content.includes('// CONTEXT_FIX_APPLIED')) {
    // Remove the entire line with actualContext
    content = content.replace(/\s*\/\/ CONTEXT_FIX_APPLIED.*\n\s*actualContext:.*,/g, '');
    console.log('✅ Removed CONTEXT_FIX_APPLIED patch');
}

// Remove the conceptsWithContext addition if it exists
content = content.replace(/\s*conceptsWithContext: conceptsArray,.*\/\/ Pass full concepts with context/g, '');

fs.writeFileSync(scholarSpherePath, content);
console.log('✅ ScholarSphere restored to original state');
console.log('\nNow restart your dev server and test if PDF upload works again.');
