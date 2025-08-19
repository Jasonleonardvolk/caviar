const path = require('path');

/** @type {import('webpack').Configuration} */
module.exports = {
  mode: 'none', // Set to 'production' for minification
  target: 'node', // VS Code extensions run in a Node.js environment
  entry: './src/extension.js', // The entry point of your extension
  output: {
    path: path.resolve(__dirname, 'dist'), // Output directory
    filename: 'extension.js',
    libraryTarget: 'commonjs2', // CommonJS compatible output
    devtoolModuleFilenameTemplate: '../[resource-path]'
  },
  externals: {
    vscode: 'commonjs vscode' // Avoid bundling vscode, which is available in the environment
  },
  resolve: {
    extensions: ['.js'] // Extensions to resolve
  },
  module: {
    rules: []
  },
  optimization: {
    minimize: false // No minification in development, set to true for production
  },
  devtool: 'nosources-source-map', // Generate source maps for debugging
  infrastructureLogging: {
    level: 'log' // Enable infrastructure logging
  }
};
