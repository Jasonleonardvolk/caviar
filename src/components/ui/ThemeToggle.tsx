/**
 * ThemeToggle.tsx - Dark ↔ Light mode switcher for TORI
 * Persistent theme management with smooth transitions
 */

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';

interface ThemeToggleProps {
  className?: string;
  size?: 'sm' | 'md' | 'lg';
  showLabel?: boolean;
}

type Theme = 'light' | 'dark';

const ThemeToggle: React.FC<ThemeToggleProps> = ({
  className = '',
  size = 'md',
  showLabel = true
}) => {
  const [theme, setTheme] = useState<Theme>('light');
  const [isLoaded, setIsLoaded] = useState(false);

  // Load theme from localStorage on mount
  useEffect(() => {
    const savedTheme = localStorage.getItem('tori-theme') as Theme;
    const systemTheme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    const initialTheme = savedTheme || systemTheme;
    
    setTheme(initialTheme);
    applyTheme(initialTheme);
    setIsLoaded(true);
  }, []);

  // Apply theme to document
  const applyTheme = (newTheme: Theme) => {
    const root = document.documentElement;
    
    if (newTheme === 'dark') {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
    
    // Also update any TORI app containers
    const toriApps = document.querySelectorAll('.tori-app');
    toriApps.forEach(app => {
      if (newTheme === 'dark') {
        app.classList.add('dark');
      } else {
        app.classList.remove('dark');
      }
    });
  };

  // Toggle theme
  const toggleTheme = () => {
    const newTheme = theme === 'light' ? 'dark' : 'light';
    setTheme(newTheme);
    applyTheme(newTheme);
    localStorage.setItem('tori-theme', newTheme);
    
    // Emit theme change event for other components
    window.dispatchEvent(new CustomEvent('tori-theme-change', {
      detail: { theme: newTheme }
    }));
  };

  // Size configurations
  const sizeConfig = {
    sm: {
      container: 'w-12 h-6',
      button: 'w-10 h-4',
      circle: 'w-4 h-4',
      icon: 'w-3 h-3',
      text: 'text-xs'
    },
    md: {
      container: 'w-14 h-7',
      button: 'w-12 h-5',
      circle: 'w-5 h-5',
      icon: 'w-3.5 h-3.5',
      text: 'text-sm'
    },
    lg: {
      container: 'w-16 h-8',
      button: 'w-14 h-6',
      circle: 'w-6 h-6',
      icon: 'w-4 h-4',
      text: 'text-base'
    }
  };

  const config = sizeConfig[size];

  if (!isLoaded) {
    return (
      <div className={`flex items-center space-x-2 ${className}`}>
        <div className={`${config.container} bg-gray-300 rounded-full animate-pulse`}></div>
        {showLabel && <div className="w-16 h-4 bg-gray-300 rounded animate-pulse"></div>}
      </div>
    );
  }

  return (
    <div className={`flex items-center space-x-3 ${className}`}>
      {/* Toggle Switch */}
      <button
        onClick={toggleTheme}
        className={`
          relative ${config.container} 
          bg-gradient-to-r 
          ${theme === 'light' 
            ? 'from-blue-400 to-cyan-400' 
            : 'from-purple-600 to-indigo-600'
          }
          rounded-full transition-all duration-300 ease-in-out
          hover:shadow-lg focus:outline-none focus:ring-2 focus:ring-offset-2
          ${theme === 'light' 
            ? 'focus:ring-cyan-400' 
            : 'focus:ring-purple-500'
          }
        `}
        aria-label={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
        title={`Currently ${theme} mode. Click to switch to ${theme === 'light' ? 'dark' : 'light'} mode.`}
      >
        {/* Background Icons */}
        <div className="absolute inset-0 flex items-center justify-between px-1">
          {/* Sun Icon */}
          <motion.div
            animate={{ 
              opacity: theme === 'light' ? 0.3 : 0.7,
              scale: theme === 'light' ? 0.8 : 1
            }}
            transition={{ duration: 0.3 }}
            className="text-yellow-300"
          >
            <svg className={config.icon} fill="currentColor" viewBox="0 0 20 20">
              <path
                fillRule="evenodd"
                d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.706.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 100-2h-1a1 1 0 100 2h1zm-7 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.05 6.464A1 1 0 106.465 5.05l-.708-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1 0 100-2H3a1 1 0 000 2h1z"
                clipRule="evenodd"
              />
            </svg>
          </motion.div>
          
          {/* Moon Icon */}
          <motion.div
            animate={{ 
              opacity: theme === 'dark' ? 0.3 : 0.7,
              scale: theme === 'dark' ? 0.8 : 1
            }}
            transition={{ duration: 0.3 }}
            className="text-slate-300"
          >
            <svg className={config.icon} fill="currentColor" viewBox="0 0 20 20">
              <path d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z" />
            </svg>
          </motion.div>
        </div>

        {/* Toggle Circle */}
        <motion.div
          animate={{
            x: theme === 'light' ? 2 : `calc(100% - ${config.circle.split(' ')[1].replace('h-', '').replace('[', '').replace(']', '')} - 2px)`,
          }}
          transition={{ 
            type: "spring", 
            stiffness: 500, 
            damping: 30,
            duration: 0.3
          }}
          className={`
            absolute top-1 ${config.circle}
            bg-white rounded-full shadow-lg
            flex items-center justify-center
          `}
        >
          {/* Current mode icon in circle */}
          <motion.div
            animate={{ rotate: theme === 'light' ? 0 : 180 }}
            transition={{ duration: 0.3 }}
            className={`${config.icon} ${theme === 'light' ? 'text-yellow-500' : 'text-purple-600'}`}
          >
            {theme === 'light' ? (
              <svg fill="currentColor" viewBox="0 0 20 20">
                <path
                  fillRule="evenodd"
                  d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.706.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 100-2h-1a1 1 0 100 2h1zm-7 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.05 6.464A1 1 0 106.465 5.05l-.708-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1 0 100-2H3a1 1 0 000 2h1z"
                  clipRule="evenodd"
                />
              </svg>
            ) : (
              <svg fill="currentColor" viewBox="0 0 20 20">
                <path d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z" />
              </svg>
            )}
          </motion.div>
        </motion.div>
      </button>

      {/* Label */}
      {showLabel && (
        <motion.span
          animate={{ opacity: 1 }}
          className={`${config.text} font-medium text-gray-700 dark:text-gray-300 transition-colors duration-300`}
        >
          {theme === 'light' ? 'Light' : 'Dark'}
        </motion.span>
      )}
    </div>
  );
};

export default ThemeToggle;

// Hook for other components to use theme state
export const useTheme = () => {
  const [theme, setTheme] = useState<Theme>('light');

  useEffect(() => {
    // Get initial theme
    const savedTheme = localStorage.getItem('tori-theme') as Theme;
    const systemTheme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    const initialTheme = savedTheme || systemTheme;
    setTheme(initialTheme);

    // Listen for theme changes
    const handleThemeChange = (e: CustomEvent) => {
      setTheme(e.detail.theme);
    };

    window.addEventListener('tori-theme-change', handleThemeChange as EventListener);
    
    return () => {
      window.removeEventListener('tori-theme-change', handleThemeChange as EventListener);
    };
  }, []);

  const toggleTheme = () => {
    const newTheme = theme === 'light' ? 'dark' : 'light';
    setTheme(newTheme);
    localStorage.setItem('tori-theme', newTheme);
    
    // Apply to document
    if (newTheme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
    
    // Emit event
    window.dispatchEvent(new CustomEvent('tori-theme-change', {
      detail: { theme: newTheme }
    }));
  };

  return { theme, toggleTheme };
};