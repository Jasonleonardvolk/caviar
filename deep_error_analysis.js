const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

console.log('Analyzing TypeScript errors...\n');

// Run tsc and capture all output
let output;
try {
    output = execSync('npx tsc --noEmit', { 
        encoding: 'utf8', 
        stdio: 'pipe',
        maxBuffer: 10 * 1024 * 1024 // 10MB buffer
    });
} catch (e) {
    output = e.stdout || '' + e.stderr || '';
}

// Write full output to file
fs.writeFileSync('full_typescript_output.txt', output);

// Parse errors
const lines = output.split('\n');
const errors = [];
const errorsByFile = {};
const errorsByCode = {};
const errorsByDirectory = {};

for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const match = line.match(/(.+?):(\d+):(\d+) - error (TS\d+): (.+)/);
    
    if (match) {
        const [_, file, lineNum, col, code, message] = match;
        const error = { file, lineNum: parseInt(lineNum), col: parseInt(col), code, message };
        
        errors.push(error);
        
        // Group by file
        if (!errorsByFile[file]) errorsByFile[file] = [];
        errorsByFile[file].push(error);
        
        // Count by error code
        errorsByCode[code] = (errorsByCode[code] || 0) + 1;
        
        // Group by directory
        const dir = file.split(/[\/\\]/)[0];
        if (!errorsByDirectory[dir]) errorsByDirectory[dir] = 0;
        errorsByDirectory[dir]++;
    }
}

console.log('=' .repeat(70));
console.log('TYPESCRIPT ERROR ANALYSIS REPORT');
console.log('=' .repeat(70));

console.log(`\nTotal: ${errors.length} errors in ${Object.keys(errorsByFile).length} files\n`);

// Show errors by directory
console.log('ERRORS BY DIRECTORY:');
console.log('-'.repeat(40));
Object.entries(errorsByDirectory)
    .sort((a, b) => b[1] - a[1])
    .forEach(([dir, count]) => {
        console.log(`  ${dir.padEnd(30)} ${count} errors`);
    });

// Show top error codes
console.log('\nTOP ERROR CODES:');
console.log('-'.repeat(40));
Object.entries(errorsByCode)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10)
    .forEach(([code, count]) => {
        console.log(`  ${code}: ${count} errors`);
        // Show what this error code means
        const sampleError = errors.find(e => e.code === code);
        if (sampleError) {
            console.log(`    Example: "${sampleError.message.substring(0, 80)}..."`);
        }
    });

// Show files with most errors
console.log('\nFILES WITH MOST ERRORS:');
console.log('-'.repeat(40));
Object.entries(errorsByFile)
    .sort((a, b) => b[1].length - a[1].length)
    .slice(0, 15)
    .forEach(([file, fileErrors]) => {
        console.log(`  ${file}`);
        console.log(`    ${fileErrors.length} errors`);
        // Show unique error codes for this file
        const codes = [...new Set(fileErrors.map(e => e.code))];
        console.log(`    Error codes: ${codes.join(', ')}`);
    });

// Check for patterns
console.log('\nPATTERN ANALYSIS:');
console.log('-'.repeat(40));

// Check for JS files
const jsFiles = Object.keys(errorsByFile).filter(f => f.endsWith('.js'));
if (jsFiles.length > 0) {
    console.log(`  JavaScript files with errors: ${jsFiles.length}`);
    jsFiles.slice(0, 5).forEach(f => console.log(`    - ${f}`));
}

// Check for test files
const testFiles = Object.keys(errorsByFile).filter(f => 
    f.includes('test') || f.includes('spec') || f.includes('example')
);
if (testFiles.length > 0) {
    console.log(`\n  Test/Example files with errors: ${testFiles.length}`);
    testFiles.slice(0, 5).forEach(f => console.log(`    - ${f}`));
}

// Check for backup files that might still be there
const backupPatterns = ['.backup', '.bak', '_backup', '_temp', '_old', '.deprecated', '_fixed'];
const backupFiles = Object.keys(errorsByFile).filter(f => 
    backupPatterns.some(pattern => f.includes(pattern))
);
if (backupFiles.length > 0) {
    console.log(`\n  WARNING: Backup files still being checked: ${backupFiles.length}`);
    backupFiles.forEach(f => console.log(`    - ${f}`));
}

// Module not found errors
const moduleErrors = errors.filter(e => e.message.includes('Cannot find module'));
if (moduleErrors.length > 0) {
    const modules = new Set();
    moduleErrors.forEach(e => {
        const match = e.message.match(/'([^']+)'/);
        if (match) modules.add(match[1]);
    });
    console.log(`\n  Missing modules: ${modules.size} unique modules`);
    Array.from(modules).slice(0, 10).forEach(m => console.log(`    - ${m}`));
}

// Save detailed report
const report = {
    timestamp: new Date().toISOString(),
    summary: {
        total_errors: errors.length,
        total_files: Object.keys(errorsByFile).length,
        directories: errorsByDirectory,
        top_error_codes: Object.entries(errorsByCode)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 10)
            .map(([code, count]) => ({ code, count }))
    },
    patterns: {
        js_files: jsFiles.length,
        test_files: testFiles.length,
        backup_files: backupFiles.length
    },
    sample_errors: errors.slice(0, 20).map(e => ({
        file: e.file,
        code: e.code,
        message: e.message
    }))
};

fs.writeFileSync('typescript_error_analysis.json', JSON.stringify(report, null, 2));

console.log('\n' + '='.repeat(70));
console.log('RECOMMENDATIONS:');
console.log('='.repeat(70));

if (jsFiles.length > 0) {
    console.log('\n1. JavaScript files are still being checked!');
    console.log('   Update tsconfig.json to exclude .js files or move them.');
}

if (backupFiles.length > 0) {
    console.log('\n2. Backup files found in source!');
    console.log('   Move these to _typescript_backups directory.');
}

if (moduleErrors.length > 50) {
    console.log('\n3. Many missing module errors.');
    console.log('   Consider installing missing types or creating declarations.');
}

console.log('\nFull error output saved to: full_typescript_output.txt');
console.log('Analysis report saved to: typescript_error_analysis.json');
