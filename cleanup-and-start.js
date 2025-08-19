const MCPProcessManager = require('./mcp-process-manager');
const MCPServerWrapper = require('./mcp-server-wrapper');
const MCPMonitor = require('./mcp-monitor');
const path = require('path');

async function cleanupAndStart() {
    const processManager = new MCPProcessManager();
    
    console.log('🧹 ===== MCP SERVER CLEANUP AND START =====');
    console.log('🎯 This script will:');
    console.log('   1. Kill all existing node processes');
    console.log('   2. Start MCP server with proper management');
    console.log('   3. Enable monitoring and health checks');
    console.log('');
    
    try {
        // Step 1: Emergency cleanup
        console.log('🔥 Step 1: Emergency cleanup of ALL node processes...');
        await processManager.killExistingProcesses();
        
        // Wait for complete termination
        console.log('⏳ Waiting 5 seconds for complete process termination...');
        await new Promise(resolve => setTimeout(resolve, 5000));
        
        // Step 2: Verify cleanup
        console.log('🔍 Step 2: Verifying cleanup...');
        const monitor = new MCPMonitor();
        const initialStatus = await monitor.checkDuplicates();
        
        if (initialStatus.totalNodeProcesses > 0) {
            console.log(`⚠️  Warning: ${initialStatus.totalNodeProcesses} node processes still running`);
            console.log('🔄 Attempting additional cleanup...');
            await processManager.killExistingProcesses();
            await new Promise(resolve => setTimeout(resolve, 2000));
        } else {
            console.log('✅ All node processes successfully terminated');
        }
        
        // Step 3: Determine server path
        const possibleServers = [
            path.join(__dirname, 'index.js'),
            path.join(__dirname, 'server.js'),
            path.join(__dirname, '+server.ts'),
            path.join(__dirname, 'mcp', 'index.js'),
            path.join(__dirname, 'backend', 'server.js')
        ];
        
        let serverPath = null;
        const fs = require('fs');
        
        for (const testPath of possibleServers) {
            if (fs.existsSync(testPath)) {
                serverPath = testPath;
                break;
            }
        }
        
        if (!serverPath) {
            console.log('⚠️  No MCP server file detected. Using placeholder path.');
            console.log('🔧 You can specify the server path as an argument:');
            console.log('   node cleanup-and-start.js path/to/your/server.js');
            serverPath = process.argv[2] || path.join(__dirname, 'index.js');
        }
        
        console.log(`🎯 Selected server: ${serverPath}`);
        
        // Step 4: Start server with management
        console.log('\n🚀 Step 3: Starting MCP server with proper management...');
        const serverWrapper = new MCPServerWrapper({
            serverPath: serverPath,
            maxRestarts: 3,
            restartDelay: 5000,
            enableHealthCheck: true,
            healthCheckInterval: 30000
        });
        
        const startSuccess = await serverWrapper.start();
        
        if (startSuccess) {
            console.log('\n✅ ===== SUCCESS =====');
            console.log('🎉 MCP Server is now running with proper process management!');
            console.log('📊 Health monitoring is active');
            console.log('🔒 Process lock file created to prevent duplicates');
            console.log('💾 PID file saved for tracking');
            console.log('📝 Logs being written to mcp-server.log');
            console.log('');
            console.log('🔧 Controls:');
            console.log('   • Press Ctrl+C to stop the server gracefully');
            console.log('   • Run "node mcp-monitor.js once" for status check');
            console.log('   • Run "node mcp-monitor.js" for continuous monitoring');
            console.log('   • Use EMERGENCY_KILL_ALL_NODE.bat for emergency shutdown');
            console.log('');
            
            // Start monitoring in background
            console.log('🖥️  Starting background monitoring...');
            setTimeout(() => {
                monitor.monitor(60000); // Check every minute
            }, 5000);
            
        } else {
            console.error('\n❌ ===== FAILED =====');
            console.error('💥 Failed to start MCP server');
            console.error('🔧 Try running EMERGENCY_KILL_ALL_NODE.bat first');
            console.error('📝 Check mcp-server.log for details');
            process.exit(1);
        }
        
    } catch (error) {
        console.error('\n💥 Critical error during cleanup and start:', error);
        console.error('🚨 Run EMERGENCY_KILL_ALL_NODE.bat to clean up manually');
        process.exit(1);
    }
}

// Handle command line arguments
if (require.main === module) {
    const command = process.argv[2];
    
    if (command === '--help' || command === '-h') {
        console.log('🔧 MCP Server Cleanup and Start Tool');
        console.log('');
        console.log('Usage:');
        console.log('  node cleanup-and-start.js [server-path]');
        console.log('  node cleanup-and-start.js --help');
        console.log('');
        console.log('Examples:');
        console.log('  node cleanup-and-start.js');
        console.log('  node cleanup-and-start.js ./my-server.js');
        console.log('  node cleanup-and-start.js ./mcp/index.js');
        console.log('');
        console.log('This tool will:');
        console.log('  1. Kill all existing node processes');
        console.log('  2. Start your MCP server with process management');
        console.log('  3. Enable health monitoring and duplicate prevention');
        process.exit(0);
    }
    
    cleanupAndStart().catch(error => {
        console.error('💥 Unexpected error:', error);
        process.exit(1);
    });
}

module.exports = { cleanupAndStart };
