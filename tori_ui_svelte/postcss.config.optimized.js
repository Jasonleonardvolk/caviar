/** 
 * Optimized PostCSS Configuration for TORI UI
 * 
 * This configuration is designed to be bulletproof and handle
 * various edge cases that can cause PostCSS parsing errors.
 */

import tailwindcss from 'tailwindcss';
import autoprefixer from 'autoprefixer';

export default {
  plugins: [
    tailwindcss({
      // Explicitly point to config file
      config: './tailwind.config.js'
    }),
    autoprefixer({
      // Browser support configuration
      overrideBrowserslist: [
        '> 1%',
        'last 2 versions',
        'not dead',
        'not ie 11'
      ]
    })
  ],
  // Parser options for better error handling
  parser: 'postcss-safe-parser',
  map: {
    inline: false,
    annotation: true
  }
};
