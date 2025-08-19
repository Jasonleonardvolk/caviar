/**
 * ELFIN Language Support Extension Installer
 * 
 * This script handles the installation of the ELFIN Language Support extension
 * by setting up the necessary files and dependencies without requiring a full
 * VSIX package.
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// Print banner
console.log('='.repeat(70));
console.log('  ELFIN Language Support Extension Installer');
console.log('='.repeat(70));
console.log('');

// Define paths
const extensionDir = __dirname;
const extensionName = 'elfin-language-support';

try {
  // Install dependencies if needed
  if (!fs.existsSync(path.join(extensionDir, 'node_modules'))) {
    console.log('Installing dependencies...');
    execSync('npm install', { cwd: extensionDir, stdio: 'inherit' });
    console.log('Dependencies installed successfully.');
  } else {
    console.log('Dependencies already installed.');
  }
  
  // Run ESLint check and fix simple issues
  console.log('Running linter...');
  try {
    execSync('npm run lint -- --fix', { cwd: extensionDir, stdio: 'inherit' });
    console.log('Linting successful.');
  } catch (error) {
    console.warn('Linting reported issues but continuing installation...');
  }
  
  // Determine VS Code extensions directory
  let extensionsDir;
  if (process.platform === 'win32') {
    extensionsDir = path.join(process.env.USERPROFILE, '.vscode', 'extensions');
  } else if (process.platform === 'darwin') {
    extensionsDir = path.join(process.env.HOME, '.vscode', 'extensions');
  } else {
    extensionsDir = path.join(process.env.HOME, '.vscode', 'extensions');
  }
  
  // Create destination directory
  const destDir = path.join(extensionsDir, `${extensionName}-0.1.0`);
  
  if (!fs.existsSync(extensionsDir)) {
    console.error(`VS Code extensions directory not found: ${extensionsDir}`);
    console.error('Please make sure VS Code is installed correctly.');
    process.exit(1);
  }
  
  // Remove existing extension if it exists
  if (fs.existsSync(destDir)) {
    console.log('Removing existing extension...');
    fs.rmSync(destDir, { recursive: true, force: true });
  }
  
  // Create extension directory
  console.log(`Installing extension to: ${destDir}`);
  fs.mkdirSync(destDir, { recursive: true });
  
  // Copy required files
  const filesToCopy = [
    'extension.js',
    'package.json',
    'language-configuration.json',
    'README.md',
    'syntaxes'
  ];
  
  for (const file of filesToCopy) {
    const sourcePath = path.join(extensionDir, file);
    const destPath = path.join(destDir, file);
    
    if (fs.statSync(sourcePath).isDirectory()) {
      // Copy directory
      fs.mkdirSync(destPath, { recursive: true });
      
      const files = fs.readdirSync(sourcePath);
      for (const subFile of files) {
        const sourceSubPath = path.join(sourcePath, subFile);
        const destSubPath = path.join(destPath, subFile);
        fs.copyFileSync(sourceSubPath, destSubPath);
      }
    } else {
      // Copy file
      fs.copyFileSync(sourcePath, destPath);
    }
  }
  
  // Create node_modules directory in the extension folder
  console.log('Setting up dependencies...');
  const destNodeModules = path.join(destDir, 'node_modules');
  fs.mkdirSync(destNodeModules, { recursive: true });
  
  // Copy only required dependencies to reduce size
  const requiredDeps = ['vscode-languageclient'];
  for (const dep of requiredDeps) {
    const sourceDep = path.join(extensionDir, 'node_modules', dep);
    const destDep = path.join(destNodeModules, dep);
    
    if (fs.existsSync(sourceDep)) {
      fs.cpSync(sourceDep, destDep, { recursive: true });
    } else {
      console.warn(`Warning: Dependency '${dep}' not found in node_modules.`);
    }
  }
  
  console.log('');
  console.log('='.repeat(70));
  console.log('  ELFIN Language Support Extension installed successfully!');
  console.log('  Please restart VS Code to activate the extension.');
  console.log('='.repeat(70));
  console.log('');
  
} catch (error) {
  console.error('Error during installation:');
  console.error(error);
  process.exit(1);
}
