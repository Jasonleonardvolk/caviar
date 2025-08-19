import { execSync } from 'child_process';
import fs from 'fs';

export default function wgslValidatePlugin() {
  return {
    name: 'wgsl-validate-plugin',
    enforce: 'pre', // run before Vite processes the file
    handleHotUpdate({ file, server }) {
      if (file.endsWith('.wgsl')) {
        try {
          execSync(`naga "${file}"`, { stdio: 'pipe' });
          console.log(`✅ WGSL OK: ${file}`);
        } catch (err) {
          console.error(`❌ WGSL ERROR in ${file}`);
          console.error(err.stdout?.toString() || err.message);
          // Prevent Vite reload — keeps bad shader from breaking live preview
          server.ws.send({
            type: 'error',
            err: {
              message: `WGSL syntax error in ${file}`,
              stack: err.stdout?.toString() || err.message
            }
          });
        }
      }
    }
  };
}