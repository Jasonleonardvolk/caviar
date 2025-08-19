// Capacitor configuration for TORI Hologram mobile app

import { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'ai.tori.hologram',
  appName: 'TORI Hologram',
  webDir: 'dist-mobile',
  bundledWebRuntime: false,
  
  ios: {
    contentInset: 'automatic',
    preferredContentMode: 'mobile',
    allowsLinkPreview: false,
    // iOS 17+ for WebGPU support
    minVersion: '17.0'
  },
  
  android: {
    minWebViewVersion: 124, // Chrome 124+ for WebGPU
    allowMixedContent: false,
    backgroundColor: '#000000',
    // Android 13+ recommended
    minSdkVersion: 33
  },
  
  plugins: {
    SplashScreen: {
      launchShowDuration: 2000,
      launchAutoHide: true,
      backgroundColor: "#000000",
      androidSplashResourceName: "splash",
      androidScaleType: "CENTER_CROP",
      showSpinner: false,
      splashFullScreen: true,
      splashImmersive: true
    },
    
    LocalNotifications: {
      smallIcon: "ic_stat_icon",
      iconColor: "#488AFF",
      sound: "beep.wav"
    },
    
    App: {
      iosScheme: "tori-holo",
      androidScheme: "tori-holo"
    }
  },
  
  server: {
    // Development only - remove for production
    url: process.env.NODE_ENV === 'development' ? 'http://localhost:5173' : undefined,
    cleartext: process.env.NODE_ENV === 'development',
    allowNavigation: [
      "tori.ai",
      "*.tori.ai"
    ]
  }
};

export default config;
