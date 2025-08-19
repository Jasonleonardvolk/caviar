/**
 * Extension entry point for the ELFIN Language Support extension.
 */

const vscode = require('vscode');
const path = require('path');
const child_process = require('child_process');

let client = null;
let outputChannel = null;

/**
 * Activate the extension
 * @param {vscode.ExtensionContext} context - Extension context
 */
function activate(_context) {
  // Create output channel for logs
  outputChannel = vscode.window.createOutputChannel('ELFIN Language Server');
  outputChannel.appendLine('ELFIN Language Server extension activated');

  // Start the language server
  startLanguageServer(_context);

  // Register commands
  const restartCommand = vscode.commands.registerCommand('elfin.restartServer', () => {
    outputChannel.appendLine('Restarting ELFIN Language Server...');
    stopLanguageServer();
    startLanguageServer(_context);
  });

  _context.subscriptions.push(restartCommand);
}

/**
 * Start the language server process and create the language client
 * @param {vscode.ExtensionContext} _context - Extension context
 */
function startLanguageServer(_context) {
  // Get configuration
  const config = vscode.workspace.getConfiguration('elfin');
  const enabled = config.get('languageServer.enabled', true);
  
  // Don't start if disabled
  if (!enabled) {
    outputChannel.appendLine('ELFIN Language Server is disabled in settings');
    return;
  }

  // Get the server path from settings
  const serverPath = config.get('languageServer.path', 'elfin-lsp');
  
  try {
    // Start the server process
    // We can use either the elfin-lsp command or the run_elfin_lsp.py script
    let serverProcess;
    try {
      // First try running with the direct elfin-lsp command
      serverProcess = child_process.spawn(serverPath, ['run'], {
        shell: true,
        stdio: 'pipe'
      });
      
      // If we get here without error, add an event listener for exit to detect problems
      serverProcess.on('error', (error) => {
        outputChannel.appendLine(`Error starting server with direct command: ${error.message}`);
        outputChannel.appendLine('Falling back to run_elfin_lsp.py script...');
        
        try {
          // Try the fallback script
          const projectRoot = path.dirname(path.dirname(__dirname));  // Go up two directories from extension dir
          const scriptPath = path.join(projectRoot, 'run_elfin_lsp.py');
          serverProcess = child_process.spawn('python', [scriptPath], {
            shell: true,
            stdio: 'pipe'
          });
          
          // Log success
          outputChannel.appendLine('Started server with fallback script');
        } catch (fallbackError) {
          outputChannel.appendLine(`Error with fallback script: ${fallbackError.message}`);
          vscode.window.showErrorMessage('Failed to start ELFIN Language Server. See output panel for details.');
        }
      });
    } catch (error) {
      outputChannel.appendLine(`Error starting server with direct command: ${error.message}`);
      outputChannel.appendLine('Falling back to run_elfin_lsp.py script...');
      
      try {
        // Try the fallback script
        const projectRoot = path.dirname(path.dirname(__dirname));  // Go up two directories from extension dir
        const scriptPath = path.join(projectRoot, 'run_elfin_lsp.py');
        serverProcess = child_process.spawn('python', [scriptPath], {
          shell: true,
          stdio: 'pipe'
        });
        
        // Log success
        outputChannel.appendLine('Started server with fallback script');
      } catch (fallbackError) {
        outputChannel.appendLine(`Error with fallback script: ${fallbackError.message}`);
        vscode.window.showErrorMessage('Failed to start ELFIN Language Server. See output panel for details.');
      }
    }

    // Create the language client
    const LanguageClient = require('vscode-languageclient').LanguageClient;
    client = new LanguageClient(
      'elfin',
      'ELFIN Language Server',
      {
        reader: serverProcess.stdout,
        writer: serverProcess.stdin
      },
      {
        documentSelector: [{ scheme: 'file', language: 'elfin' }],
        outputChannel: outputChannel,
        synchronize: {
          configurationSection: 'elfin',
          fileEvents: vscode.workspace.createFileSystemWatcher('**/*.elfin')
        }
      }
    );

    // Handle server process events
    serverProcess.stderr.on('data', (data) => {
      outputChannel.appendLine(`[Server Error] ${data.toString()}`);
    });

    serverProcess.on('exit', (code, signal) => {
      outputChannel.appendLine(`Language server exited with code ${code} and signal ${signal}`);
      if (code !== 0) {
        vscode.window.showErrorMessage('ELFIN Language Server exited unexpectedly. See the output channel for details.');
      }
    });

    // Start the client
    client.start();
    outputChannel.appendLine('ELFIN Language Server started');
  } catch (error) {
    outputChannel.appendLine(`Error starting language server: ${error.message}`);
    vscode.window.showErrorMessage(`Failed to start ELFIN Language Server: ${error.message}`);
  }
}

/**
 * Stop the language server
 */
function stopLanguageServer() {
  if (client) {
    client.stop();
    client = null;
    outputChannel.appendLine('ELFIN Language Server stopped');
  }
}

/**
 * Deactivate the extension
 */
function deactivate() {
  stopLanguageServer();
  if (outputChannel) {
    outputChannel.appendLine('ELFIN Language Server extension deactivated');
    outputChannel.dispose();
  }
}

module.exports = {
  activate,
  deactivate
};
