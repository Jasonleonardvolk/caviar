const { execSync } = require('child_process');

console.log('Checking TypeScript errors...\n');

// Run tsc and capture errors
let output;
try {
    output = execSync('npx tsc --noEmit', { 
        encoding: 'utf8', 
        stdio: 'pipe',
        maxBuffer: 10 * 1024 * 1024
    });
    console.log('No errors found!');
    process.exit(0);
} catch (e) {
    output = e.stdout || '' + e.stderr || '';
}

// Count errors
const errorLines = output.split('\n').filter(line => line.includes('error TS'));
const totalErrors = errorLines.length;

// Count by directory
const dirCounts = {};
errorLines.forEach(line => {
    const match = line.match(/^([^:]+):/);
    if (match) {
        const path = match[1];
        const dir = path.split(/[\/\\]/)[0];
        dirCounts[dir] = (dirCounts[dir] || 0) + 1;
    }
});

console.log(`Total errors: ${totalErrors}\n`);
console.log('Errors by directory:');
Object.entries(dirCounts)
    .sort((a, b) => b[1] - a[1])
    .forEach(([dir, count]) => {
        console.log(`  ${dir}: ${count}`);
    });
