/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './src/**/*.{html,js,svelte,ts}',
    './src/**/*.svelte'
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // TORI brand colors
        tori: {
          primary: '#4f46e5',
          secondary: '#818cf8',
          accent: '#007acc',
          background: '#f9f9f9',
          surface: '#ffffff',
          text: '#111111',
          muted: '#666666'
        },
        // Memory system colors
        memory: {
          primary: '#4f46e5',
          secondary: '#818cf8',
          vault: '#059669',
          conversation: '#dc2626',
          thought: '#7c3aed'
        },
        // Ghost persona colors
        ghost: {
          mentor: '#00ffc8',
          mystic: '#bf00ff',
          chaotic: '#ff5500',
          oracular: '#ffd700',
          dreaming: '#3399ff',
          unsettled: '#888888'
        },
        // Extended color palette
        primary: {
          50: '#f0f9ff',
          100: '#e0f2fe',
          200: '#bae6fd',
          300: '#7dd3fc',
          400: '#38bdf8',
          500: '#0ea5e9',
          600: '#0284c7',
          700: '#0369a1',
          800: '#075985',
          900: '#0c4a6e'
        }
      },
      fontFamily: {
        sans: [
          '-apple-system', 
          'BlinkMacSystemFont', 
          'Segoe UI', 
          'Roboto', 
          'Oxygen', 
          'Ubuntu', 
          'Cantarell', 
          'sans-serif'
        ],
        mono: [
          'JetBrains Mono',
          'Fira Code', 
          'SF Mono', 
          'Monaco', 
          'Inconsolata', 
          'Roboto Mono', 
          'monospace'
        ]
      },
      spacing: {
        '18': '4.5rem',
        '88': '22rem'
      },
      maxWidth: {
        '8xl': '88rem',
        '9xl': '96rem'
      },
      animation: {
        'fade-in': 'fadeIn 0.3s ease-in-out',
        'slide-up': 'slideUp 0.3s ease-out',
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'shimmer': 'shimmer 2s ease-in-out infinite',
        'pulse-gentle': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'float': 'float 6s ease-in-out infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
        'ghost-shimmer': 'ghost-shimmer 3s ease-in-out infinite'
      },
      keyframes: {
        fadeIn: {
          'from': { opacity: '0' },
          'to': { opacity: '1' }
        },
        slideUp: {
          'from': { 
            opacity: '0',
            transform: 'translateY(10px)'
          },
          'to': { 
            opacity: '1',
            transform: 'translateY(0)'
          }
        },
        shimmer: {
          '0%, 100%': { opacity: '0.5' },
          '50%': { opacity: '1' }
        },
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-10px)' }
        },
        glow: {
          '0%': { boxShadow: '0 0 5px rgba(79, 70, 229, 0.3)' },
          '100%': { boxShadow: '0 0 20px rgba(79, 70, 229, 0.5)' }
        },
        'ghost-shimmer': {
          '0%, 100%': { opacity: '0.1' },
          '50%': { opacity: '0.3' }
        }
      },
      backdropBlur: {
        xs: '2px'
      }
    }
  },
  plugins: [
    // Add custom utilities
    function({ addUtilities }) {
      const newUtilities = {
        '.text-balance': {
          'text-wrap': 'balance'
        },
        '.scrollbar-thin': {
          'scrollbar-width': 'thin'
        },
        '.scrollbar-none': {
          'scrollbar-width': 'none',
          '&::-webkit-scrollbar': {
            display: 'none'
          }
        }
      }
      addUtilities(newUtilities)
    }
  ]
};