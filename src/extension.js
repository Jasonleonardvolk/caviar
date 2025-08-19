const path = require('path');
const fs = require('fs');
const vscode = require('vscode');
const { LanguageClient, TransportKind } = require('vscode-languageclient/node');

let client;
let statusBarItem;

/**
 * Activate the extension
 * @param {vscode.ExtensionContext} context Extension context
 */
function activate(context) {
  console.log('ELFIN Language Support activated');
  
  // Get extension settings
  const config = vscode.workspace.getConfiguration('elfin');
  
  // Path to the Language Server module
  const serverModule = getServerPath();
  
  // Server launch options
  const serverOptions = {
    run: {
      command: 'python',
      args: [serverModule],
      transport: TransportKind.stdio,
    },
    debug: {
      command: 'python',
      args: [serverModule, '--log-level=DEBUG'],
      transport: TransportKind.stdio,
    }
  };
  
  // Client options
  const clientOptions = {
    documentSelector: [
      { scheme: 'file', language: 'elfin' },
      { scheme: 'untitled', language: 'elfin' }
    ],
    synchronize: {
      fileEvents: vscode.workspace.createFileSystemWatcher('**/*.elfin')
    }
  };
  
  // Create and start the client
  client = new LanguageClient(
    'elfinLanguageServer',
    'ELFIN Language Server',
    serverOptions,
    clientOptions
  );
  
  // Register commands
  registerCommands(context);
  
  // Create status bar item
  createStatusBar(context);
  
  // Start the client
  client.start();
  
  // Push client to subscriptions to be disposed when extension is deactivated
  context.subscriptions.push(client);
}

/**
 * Find the path to the language server
 * @returns {string} Path to the language server module
 */
function getServerPath() {
  try {
    // Try to find the installed Python package
    const pythonPath = 'elfin_lsp.server';
    return pythonPath;
  } catch (err) {
    console.error('Failed to locate ELFIN language server module:', err);
    vscode.window.showErrorMessage('Failed to locate ELFIN language server. Make sure it is installed.');
    throw err;
  }
}

/**
 * Register extension commands
 * @param {vscode.ExtensionContext} context Extension context
 */
function registerCommands(context) {
  // Command to open documentation
  context.subscriptions.push(
    vscode.commands.registerCommand('elfin.openDocs', (systemName) => {
      // Construct the URL to the documentation
      const baseUrl = 'https://elfin-lang.github.io/docs/systems/';
      const url = `${baseUrl}${systemName || 'index'}.html`;
      
      // Open the browser
      vscode.env.openExternal(vscode.Uri.parse(url));
    })
  );
  
  // Command to run system tests
  context.subscriptions.push(
    vscode.commands.registerCommand('elfin.runSystemTests', (systemName, uri) => {
      if (!systemName) {
        vscode.window.showWarningMessage('No system name provided');
        return;
      }
      
      // Create a terminal to run tests
      const terminal = getOrCreateTerminal('ELFIN Tests');
      terminal.show();
      
      // Build the test command
      let testCommand = `pytest -v --system=${systemName}`;
      
      // If URI is provided, add it to the command
      if (uri) {
        testCommand += ` "${uri}"`;
      }
      
      // Run the command
      terminal.sendText(testCommand);
    })
  );
  
  // Command to build and run
  context.subscriptions.push(
    vscode.commands.registerCommand('elfin.buildRun', () => {
      const terminal = getOrCreateTerminal('ELFIN');
      terminal.show(true);
      terminal.sendText('elfin build && elfin run');
    })
  );
  
  // Command to scaffold new controller
  context.subscriptions.push(
    vscode.commands.registerCommand('elfin.scaffoldController', async () => {
      const name = await vscode.window.showInputBox({ 
        prompt: 'Controller name',
        placeHolder: 'MyController'
      });
      if (!name) return;
      
      // Get workspace folder
      const ws = vscode.workspace.workspaceFolders?.[0].uri.fsPath;
      if (!ws) {
        vscode.window.showErrorMessage('No workspace folder found');
        return;
      }
      
      // Create file path
      const file = path.join(ws, `${name.toLowerCase()}.elfin`);
      
      // Create file content
      const template = 
`import Helpers from "std/helpers.elfin";

system ${name} {
  continuous_state: [x, v];
  inputs: [u];
  
  params {
    m: mass[kg] = 1.0;
    k: spring_const[N/m] = 10.0;
    b: damping[N*s/m] = 0.5;
  }
  
  flow_dynamics {
    # Position derivative
    x_dot = v;
    
    # Velocity derivative
    v_dot = (-k * x - b * v + u) / m;
  }
}`;

      // Write file
      fs.writeFileSync(file, template);
      
      // Open file
      const doc = await vscode.workspace.openTextDocument(file);
      vscode.window.showTextDocument(doc);
      
      vscode.window.showInformationMessage(`Created new controller: ${name}`);
    })
  );
}

/**
 * Create status bar item
 * @param {vscode.ExtensionContext} context Extension context
 */
function createStatusBar(context) {
  // Create status bar item
  statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
  statusBarItem.command = 'elfin.buildRun';
  statusBarItem.text = '$(play) ELFIN';
  statusBarItem.tooltip = 'Build & Run active ELFIN template';
  statusBarItem.show();
  
  // Push to subscriptions
  context.subscriptions.push(statusBarItem);
  
  // Update status bar visibility based on active document
  context.subscriptions.push(
    vscode.window.onDidChangeActiveTextEditor(updateStatusBarVisibility)
  );
  
  // Initial visibility
  updateStatusBarVisibility(vscode.window.activeTextEditor);
}

/**
 * Update status bar visibility
 * @param {vscode.TextEditor} editor Active text editor
 */
function updateStatusBarVisibility(editor) {
  if (statusBarItem) {
    statusBarItem.hide();
    
    if (editor && editor.document.languageId === 'elfin') {
      statusBarItem.show();
    }
  }
}

/**
 * Get or create a terminal
 * @param {string} name Terminal name
 * @returns {vscode.Terminal} Terminal
 */
function getOrCreateTerminal(name) {
  const existing = vscode.window.terminals.find(t => t.name === name);
  return existing || vscode.window.createTerminal(name);
}

/**
 * Deactivate the extension
 */
function deactivate() {
  // Clean up status bar
  if (statusBarItem) {
    statusBarItem.dispose();
  }
  
  // Stop the client
  if (!client) {
    return undefined;
  }
  return client.stop();
}

module.exports = { activate, deactivate };
