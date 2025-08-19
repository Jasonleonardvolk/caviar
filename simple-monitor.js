import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

async function checkNodeProcesses() {
    console.log('🔍 Checking node processes...\n');
    
    try {
        const { stdout } = await execAsync('tasklist /FI "IMAGENAME eq node.exe" /FO TABLE');
        const lines = stdout.split('\n').filter(line => line.trim() && line.includes('node.exe'));
        
        if (lines.length === 0) {
            console.log('✅ No node processes running');
            return { count: 0, status: 'CLEAN' };
        }
        
        console.log(`📊 Found ${lines.length} node processes:`);
        lines.forEach((line, index) => {
            const parts = line.split(/\s+/);
            const pid = parts[1];
            const memory = parts[4];
            console.log(`  ${index + 1}. PID: ${pid}, Memory: ${memory}`);
        });
        
        if (lines.length > 1) {
            console.log('\n⚠️  WARNING: Multiple node processes detected!');
            console.log('🔥 Run: node kill-all-node-simple.js');
            return { count: lines.length, status: 'DUPLICATES' };
        } else {
            console.log('\n✅ Single node process - looks healthy');
            return { count: lines.length, status: 'HEALTHY' };
        }
        
    } catch (error) {
        console.log('✅ No node processes found');
        return { count: 0, status: 'CLEAN' };
    }
}

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
    checkNodeProcesses().catch(console.error);
}

export default checkNodeProcesses;