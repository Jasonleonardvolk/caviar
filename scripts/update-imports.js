/**
 * Codemod script to update imports from relative paths to @itori package paths
 * 
 * Usage:
 *   jscodeshift -t scripts/update-imports.js apps/**/*.{ts,tsx,jsx}
 */

module.exports = function(file, api) {
  const j = api.jscodeshift;
  const root = j(file.source);
  let hasModifications = false;

  // Map of known component/module paths to their new @itori package locations
  const importMappings = {
    // Map for runtime-bridge components
    'hooks/useAlanSocket': '@itori/runtime-bridge',
    '../hooks/useAlanSocket': '@itori/runtime-bridge',
    '../../hooks/useAlanSocket': '@itori/runtime-bridge',
    '../../../hooks/useAlanSocket': '@itori/runtime-bridge',
    'types/websocket': '@itori/runtime-bridge',
    '../types/websocket': '@itori/runtime-bridge',
    '../../types/websocket': '@itori/runtime-bridge',
    '../../../types/websocket': '@itori/runtime-bridge',
    
    // Map for UI components
    'components/WebSocketStatus': '@itori/ui-kit',
    '../components/WebSocketStatus': '@itori/ui-kit',
    '../../components/WebSocketStatus': '@itori/ui-kit',
    '../../../components/WebSocketStatus': '@itori/ui-kit',
    'components/ChatWindow': '@itori/ui-kit',
    '../components/ChatWindow': '@itori/ui-kit',
    '../../components/ChatWindow': '@itori/ui-kit',
    '../../../components/ChatWindow': '@itori/ui-kit',
    
    // Add more mappings as needed for other components
  };

  // Process import declarations
  root.find(j.ImportDeclaration).forEach(path => {
    const importPath = path.node.source.value;
    
    // Check if this import path should be updated
    for (const [oldPath, newPackage] of Object.entries(importMappings)) {
      if (importPath === oldPath) {
        // Update the import source to the new package
        path.node.source.value = newPackage;
        hasModifications = true;
        
        // Log the transformation for debugging
        console.log(`Transformed: ${importPath} -> ${newPackage}`);
        break;
      }
    }
  });

  return hasModifications ? root.toSource() : null;
};
