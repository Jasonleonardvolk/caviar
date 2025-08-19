// Simple emergency node killer - ES Module compatible
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

async function killAllNodeProcesses() {
    console.log('🚨 EMERGENCY: Killing all node processes...');
    
    try {
        // Kill all node processes
        await execAsync('taskkill /F /IM node.exe');
        console.log('💀 All node processes terminated');
    } catch (error) {
        if (error.message.includes('not found') || error.message.includes('No tasks')) {
            console.log('✅ No node processes found to kill');
        } else {
            console.log('⚠️  Error (processes may already be dead):', error.message);
        }
    }
    
    // Wait 2 seconds
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Check what's left
    try {
        const { stdout } = await execAsync('tasklist /FI "IMAGENAME eq node.exe" /FO CSV');
        const lines = stdout.split('\n').filter(line => line.includes('node.exe'));
        
        if (lines.length > 0) {
            console.log(`⚠️  Warning: ${lines.length} node processes still running`);
        } else {
            console.log('✅ SUCCESS: All node processes terminated!');
        }
    } catch (error) {
        console.log('✅ SUCCESS: No node processes found!');
    }
    
    console.log('\n🎉 Emergency cleanup completed!');
    console.log('You can now start your MCP server safely.');
}

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
    killAllNodeProcesses().catch(console.error);
}

export default killAllNodeProcesses;