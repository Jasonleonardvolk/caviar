#!/usr/bin/env node
/**
 * Fix TailwindCSS tori-button error
 */

import { promises as fs } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

async function fixToriButton() {
  console.log('🔧 Fixing TailwindCSS tori-button error...\n');

  // Step 1: Update app.css to use standard Tailwind classes
  const appCssPath = join(__dirname, 'src', 'app.css');
  let appCss = await fs.readFile(appCssPath, 'utf-8');
  
  // Replace the button component definitions with standard Tailwind utilities
  const updatedCss = appCss
    .replace(
      /\.tori-button-primary\s*{[^}]*@apply[^}]+}/g,
      `.tori-button-primary {
    /* Primary button styles */
    background-color: rgb(37 99 235); /* bg-blue-600 */
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 0.375rem;
    font-weight: 500;
    font-size: 0.875rem;
    line-height: 1.25rem;
    border: 1px solid transparent;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    transition: all 200ms;
  }
  
  .tori-button-primary:hover {
    background-color: rgb(29 78 216); /* bg-blue-700 */
  }
  
  .tori-button-primary:focus {
    outline: 2px solid transparent;
    outline-offset: 2px;
    box-shadow: 0 0 0 2px white, 0 0 0 4px rgb(59 130 246); /* focus:ring-blue-500 */
  }`
    )
    .replace(
      /\.tori-button-secondary\s*{[^}]*@apply[^}]+}/g,
      `.tori-button-secondary {
    /* Secondary button styles */
    background-color: rgb(243 244 246); /* bg-gray-100 */
    color: rgb(17 24 39); /* text-gray-900 */
    padding: 0.5rem 1rem;
    border-radius: 0.375rem;
    font-weight: 500;
    font-size: 0.875rem;
    line-height: 1.25rem;
    border: 1px solid transparent;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    transition: all 200ms;
  }
  
  .tori-button-secondary:hover {
    background-color: rgb(229 231 235); /* bg-gray-200 */
  }
  
  .tori-button-secondary:focus {
    outline: 2px solid transparent;
    outline-offset: 2px;
    box-shadow: 0 0 0 2px white, 0 0 0 4px rgb(107 114 128); /* focus:ring-gray-500 */
  }`
    );

  await fs.writeFile(appCssPath, updatedCss, 'utf-8');
  console.log('✅ Updated app.css with explicit button styles');

  // Step 2: Clean build cache
  const svelteKitPath = join(__dirname, '.svelte-kit');
  try {
    await fs.rm(svelteKitPath, { recursive: true, force: true });
    console.log('✅ Cleared .svelte-kit cache');
  } catch (e) {
    console.log('ℹ️  No cache to clear');
  }

  console.log('\n✨ Fix complete! Now run: npm run dev');
}

fixToriButton().catch(console.error);
