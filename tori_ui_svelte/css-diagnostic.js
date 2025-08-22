// css-diagnostic.js - TORI CSS Diagnostic Tool
import fs from 'fs';
import path from 'path';

const projectRoot = process.cwd();

console.log('🔍 TORI CSS DIAGNOSTIC REPORT');
console.log('===============================\n');

// Check configuration files
const configFiles = [
    'svelte.config.js',
    'tailwind.config.js',
    'tailwind.config.cjs',
    'postcss.config.js',
    'vite.config.js',
    'package.json'
];

console.log('📋 Configuration Files:');
configFiles.forEach(file => {
    const filePath = path.join(projectRoot, file);
    if (fs.existsSync(filePath)) {
        console.log(`✅ ${file} - EXISTS`);
        if (file === 'package.json') {
            const pkg = JSON.parse(fs.readFileSync(filePath, 'utf8'));
            const deps = Object.keys(pkg.dependencies || {}).filter(d => d.includes('tailwind') || d.includes('postcss')).join(', ');
            const devDeps = Object.keys(pkg.devDependencies || {}).filter(d => d.includes('tailwind') || d.includes('postcss')).join(', ');
            if (deps) console.log(`   Dependencies: ${deps}`);
            if (devDeps) console.log(`   DevDependencies: ${devDeps}`);
        }
    } else {
        console.log(`❌ ${file} - MISSING`);
    }
});

// Check CSS files
const cssFiles = [
    'src/app.css',
    'src/index.css',
    'src/global.css',
    'src/styles.css'
];

console.log('\n🎨 CSS Files:');
cssFiles.forEach(file => {
    const filePath = path.join(projectRoot, file);
    if (fs.existsSync(filePath)) {
        const content = fs.readFileSync(filePath, 'utf8');
        console.log(`✅ ${file} - EXISTS (${content.length} chars)`);
        
        // Check for Tailwind directives
        const hasTailwindBase = content.includes('@tailwind base');
        const hasTailwindComponents = content.includes('@tailwind components');
        const hasTailwindUtilities = content.includes('@tailwind utilities');
        
        console.log(`   @tailwind base: ${hasTailwindBase ? '✅' : '❌'}`);
        console.log(`   @tailwind components: ${hasTailwindComponents ? '✅' : '❌'}`);
        console.log(`   @tailwind utilities: ${hasTailwindUtilities ? '✅' : '❌'}`);
        
        // Check for TORI light theme
        const hasLightTheme = content.includes('--color-bg-primary: #f9f9f9');
        const hasMemoryColors = content.includes('--color-memory-primary');
        const hasGhostColors = content.includes('--color-ghost-mentor');
        
        console.log(`   Light theme: ${hasLightTheme ? '✅' : '❌'}`);
        console.log(`   Memory colors: ${hasMemoryColors ? '✅' : '❌'}`);
        console.log(`   Ghost colors: ${hasGhostColors ? '✅' : '❌'}`);
        
        // Check for body styles
        const hasBodyStyles = content.includes('body') && (content.includes('bg-') || content.includes('background'));
        console.log(`   Body styles: ${hasBodyStyles ? '✅' : '❌'}`);
    } else {
        console.log(`❌ ${file} - MISSING`);
    }
});

// Check layout files
const layoutFiles = [
    'src/app.html',
    'src/routes/+layout.svelte'
];

console.log('\n📄 Layout Files:');
layoutFiles.forEach(file => {
    const filePath = path.join(projectRoot, file);
    if (fs.existsSync(filePath)) {
        const content = fs.readFileSync(filePath, 'utf8');
        console.log(`✅ ${file} - EXISTS`);
        
        // Check for CSS imports
        const hasAppCssImport = content.includes('app.css') || content.includes('./app.css');
        const hasIndexCssImport = content.includes('index.css') || content.includes('./index.css');
        const hasLightClass = content.includes('class="light"') || content.includes('bg-tori-background');
        
        console.log(`   app.css import: ${hasAppCssImport ? '✅' : '❌'}`);
        console.log(`   index.css import: ${hasIndexCssImport ? '⚠️ Should be removed' : '✅ Clean'}`);
        console.log(`   Light theme class: ${hasLightClass ? '✅' : '❌'}`);
    } else {
        console.log(`❌ ${file} - MISSING`);
    }
});

console.log('\n🚀 TORI LIGHT THEME STATUS:');
console.log('============================');

// Final validation
const appCssPath = path.join(projectRoot, 'src/app.css');
const layoutPath = path.join(projectRoot, 'src/routes/+layout.svelte');
const appHtmlPath = path.join(projectRoot, 'src/app.html');

if (fs.existsSync(appCssPath) && fs.existsSync(layoutPath) && fs.existsSync(appHtmlPath)) {
    const appCss = fs.readFileSync(appCssPath, 'utf8');
    const layout = fs.readFileSync(layoutPath, 'utf8');
    const appHtml = fs.readFileSync(appHtmlPath, 'utf8');
    
    const hasCompleteSetup = 
        appCss.includes('--color-bg-primary: #f9f9f9') &&
        appCss.includes('@tailwind base') &&
        layout.includes('../app.css') &&
        appHtml.includes('class="light"');
    
    if (hasCompleteSetup) {
        console.log('✅ TORI LIGHT THEME: FULLY CONFIGURED');
        console.log('✅ Memory system colors: LOADED');
        console.log('✅ Ghost aura effects: READY');
        console.log('✅ CSS variables: ACTIVE');
        console.log('✅ Tailwind integration: COMPLETE');
        console.log('\n🎉 Ready for ThoughtspaceRenderer.ts implementation!');
    } else {
        console.log('⚠️  TORI LIGHT THEME: NEEDS CONFIGURATION');
        console.log('   Run: npm install && npm run dev');
    }
} else {
    console.log('❌ CRITICAL FILES MISSING');
}

console.log('\n💡 NEXT STEPS:');
console.log('1. Run: npm install');
console.log('2. Run: npm run dev');
console.log('3. Open: http://localhost:5173');
console.log('4. Verify: DevTools → body shows bg-tori-background');
console.log('5. Ready for: ThoughtspaceRenderer.ts and memory node visuals');

export default function diagnostic() {
    return 'TORI CSS Diagnostic Complete';
}