#!/usr/bin/env node
/**
 * Update script to wire up the holographic system
 * This replaces old GhostEngine imports with the real implementation
 */

const fs = require('fs');
const path = require('path');

console.log('🔧 TORI Holographic System Wiring Script');
console.log('=======================================\n');

const updates = [
    {
        name: 'Update App.svelte imports',
        file: 'tori_ui_svelte/src/App.svelte',
        replacements: [
            {
                from: "import { GhostEngine } from './lib/ghostEngine.js'",
                to: "import { GhostEngine } from './lib/holographicBridge.js'"
            },
            {
                from: "from './lib/ghostEngine'",
                to: "from './lib/holographicBridge.js'"
            }
        ]
    },
    {
        name: 'Update component imports',
        file: 'tori_ui_svelte/src/lib/components/HolographicVisualization.svelte',
        replacements: [
            {
                from: "import { ToriHolographicRenderer } from '../lib/holographicRenderer';",
                to: "import { RealGhostEngine } from '../lib/realGhostEngine.js';"
            }
        ]
    },
    {
        name: 'Create .env for display configuration',
        file: 'tori_ui_svelte/.env',
        content: `# TORI Holographic Display Configuration
VITE_DISPLAY_TYPE=webgpu_only
# Options: looking_glass_portrait, looking_glass_32, looking_glass_65, webgpu_only

# WebSocket Backend
VITE_WS_URL=ws://localhost:8765/audio_stream

# Enable features
VITE_ENABLE_AUDIO=true
VITE_ENABLE_HOTT=true
VITE_ENABLE_DEBUG=true
`
    },
    {
        name: 'Update vite.config.js for WGSL support',
        file: 'tori_ui_svelte/vite.config.js',
        append: `
// Add WGSL file support
export default defineConfig({
  // ... existing config ...
  assetsInclude: ['**/*.wgsl'],
  server: {
    fs: {
      allow: ['..'] // Allow access to parent directory for shaders
    }
  }
});
`
    }
];

// Process updates
updates.forEach(update => {
    console.log(`📝 ${update.name}...`);
    
    try {
        const filePath = path.join(__dirname, update.file);
        
        if (update.content) {
            // Write new file
            fs.writeFileSync(filePath, update.content);
            console.log(`   ✅ Created ${update.file}`);
        } else if (update.replacements) {
            // Update existing file
            if (fs.existsSync(filePath)) {
                let content = fs.readFileSync(filePath, 'utf8');
                
                update.replacements.forEach(replacement => {
                    if (content.includes(replacement.from)) {
                        content = content.replace(replacement.from, replacement.to);
                        console.log(`   ✅ Replaced import in ${update.file}`);
                    }
                });
                
                fs.writeFileSync(filePath, content);
            } else {
                console.log(`   ⚠️  File not found: ${update.file}`);
            }
        } else if (update.append) {
            // Append to file
            if (fs.existsSync(filePath)) {
                const content = fs.readFileSync(filePath, 'utf8');
                if (!content.includes('assetsInclude')) {
                    fs.appendFileSync(filePath, update.append);
                    console.log(`   ✅ Updated ${update.file}`);
                }
            }
        }
    } catch (error) {
        console.log(`   ❌ Error: ${error.message}`);
    }
});

// Create symbolic link for backward compatibility
console.log('\n🔗 Creating compatibility links...');
try {
    const oldPath = path.join(__dirname, 'tori_ui_svelte/src/lib/ghostEngine.js');
    const newPath = path.join(__dirname, 'tori_ui_svelte/src/lib/holographicBridge.js');
    
    if (fs.existsSync(oldPath)) {
        fs.renameSync(oldPath, oldPath + '.backup');
        console.log('   ✅ Backed up old ghostEngine.js');
    }
    
    // Create a redirect file
    fs.writeFileSync(oldPath, `// Redirect to the real implementation
export * from './holographicBridge.js';
console.log('🔄 Redirecting to connected holographic system...');
`);
    console.log('   ✅ Created redirect from ghostEngine.js to holographicBridge.js');
    
} catch (error) {
    console.log(`   ⚠️  Could not create redirect: ${error.message}`);
}

console.log('\n✨ Wiring complete! Your holographic system is now connected!');
console.log('\n📚 Next steps:');
console.log('1. Run: python start_hologram_system.py');
console.log('2. Or manually:');
console.log('   - Terminal 1: python audio_hologram_bridge.py');
console.log('   - Terminal 2: cd tori_ui_svelte && npm run dev');
console.log('3. Open http://localhost:5173 in a WebGPU-enabled browser');
console.log('\n🎉 Enjoy your holographic experience!');
