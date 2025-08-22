/**
 * Ghost System Type Definitions
 * Defines the structure for Ghost personas and their capabilities
 */

export interface DimensionalValues {
  sarcasm: number;        // 0-1: Dry wit level
  formality: number;      // 0-1: Professional vs casual
  assertiveness: number;  // 0-1: Confidence level
  creativity: number;     // 0-1: Creative expression
  empathy: number;        // 0-1: Understanding level
  humor: number;          // 0-1: Humor appreciation
  enthusiasm: number;     // 0-1: Energy level
  patience: number;       // 0-1: Tolerance level
  curiosity: number;      // 0-1: Exploration drive
  precision: number;      // 0-1: Accuracy focus
}

export interface ConversationStyle {
  greeting: string;
  vocabulary: string[];
  toneMarkers: string[];
}

export interface ResponsePatterns {
  analytical: number;
  intuitive: number;
  supportive: number;
  challenging: number;
  educational: number;
}

export interface CognitiveStyle {
  holisticVsAnalytical: number;
  intuitiveVsMethodical: number;
  visualVsVerbal: number;
  theoreticalVsPractical: number;
}

export interface GhostInteractions {
  synthesizes: string[];
  amplifies: string[];
  harmonizes: string[];
  bridges: string[];
}

export interface RenderStyle {
  colors: {
    primary: string;
    secondary: string;
    accent: string;
    glow: string;
  };
  animation: string;
  particleEffect: string;
  hologramStyle: string;
}

export interface GhostPersona {
  id: string;
  name: string;
  title: string;
  description: string;
  dimensions: DimensionalValues;
  traits: string[];
  expertise: string[];
  background: string;
  conversationStyle: ConversationStyle;
  responsePatterns: ResponsePatterns;
  interests: string[];
  cognitiveStyle: CognitiveStyle;
  specialAbilities: string[];
  ghostInteractions: GhostInteractions;
  activationTriggers: string[];
  systemRole: string;
  renderStyle: RenderStyle;
}