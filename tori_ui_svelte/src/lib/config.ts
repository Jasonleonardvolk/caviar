// Environment Configuration
export const config = {
  // API Configuration
  API_BASE_URL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:3001',
  WS_URL: import.meta.env.VITE_WS_URL || 'ws://localhost:3001',
  
  // Feature Flags
  ENABLE_WEBSOCKETS: import.meta.env.VITE_ENABLE_WEBSOCKETS !== 'false',
  ENABLE_FALLBACK_MODE: import.meta.env.VITE_ENABLE_FALLBACK_MODE !== 'false',
  ENABLE_DEBUG_LOGGING: import.meta.env.VITE_DEBUG === 'true',
  
  // TORI System Configuration
  DEFAULT_COHERENCE: parseFloat(import.meta.env.VITE_DEFAULT_COHERENCE || '0.8'),
  DEFAULT_ENTROPY: parseFloat(import.meta.env.VITE_DEFAULT_ENTROPY || '0.2'),
  MAX_CONVERSATION_HISTORY: parseInt(import.meta.env.VITE_MAX_HISTORY || '100'),
  
  // Thoughtspace Configuration
  THOUGHTSPACE_ENABLED: import.meta.env.VITE_THOUGHTSPACE_ENABLED !== 'false',
  MAX_CONCEPT_NODES: parseInt(import.meta.env.VITE_MAX_NODES || '50'),
  
  // Ghost Persona Configuration
  GHOST_EMERGENCE_THRESHOLD: parseFloat(import.meta.env.VITE_GHOST_THRESHOLD || '0.3'),
  AUTO_GHOST_MORPHING: import.meta.env.VITE_AUTO_GHOST_MORPH !== 'false',
  
  // Development
  DEV_MODE: import.meta.env.DEV,
  PROD_MODE: import.meta.env.PROD
};

// Runtime environment detection
export const isProduction = config.PROD_MODE;
export const isDevelopment = config.DEV_MODE;

// Logging utility that respects debug flag
export function debugLog(...args: any[]) {
  if (config.ENABLE_DEBUG_LOGGING || isDevelopment) {
    console.log('[TORI DEBUG]', ...args);
  }
}

// API URL resolver
export function resolveApiUrl(endpoint: string): string {
  return `${config.API_BASE_URL}${endpoint}`;
}

// WebSocket URL resolver
export function resolveWsUrl(): string {
  return config.WS_URL;
}