import { GhostPersona } from '$lib/stores/ghostPersona';
// Extension to ghostPersona.ts for image interpretation

export interface ImageInterpretation {
  insight: string;
  concepts: string[];
  emotionalResonance: {
    excitement: number;
    calmness: number;
    energy: number;
    clarity: number;
  };
  morphStyle: 'explosive' | 'flowing' | 'structured' | 'organic' | 'crystalline';
}

// Extend GhostPersona class with image analysis
export class GhostPersonaExtended extends GhostPersona {
  constructor(name: string) {
    super(name);
  }
  
  async analyzeImage(imageData: ImageData): Promise<string> {
    // Each Ghost interprets images differently
    const brightness = this.calculateBrightness(imageData);
    const complexity = this.calculateComplexity(imageData);
    const warmth = this.calculateWarmth(imageData);
    
    switch (this.name) {
      case 'Mentor':
        return this.mentorImageAnalysis(brightness, complexity, warmth);
      case 'Scholar':
        return this.scholarImageAnalysis(brightness, complexity, warmth);
      case 'Explorer':
        return this.explorerImageAnalysis(brightness, complexity, warmth);
      case 'Architect':
        return this.architectImageAnalysis(brightness, complexity, warmth);
      case 'Creator':
        return this.creatorImageAnalysis(brightness, complexity, warmth);
      default:
        return "I see potential waiting to be transformed...";
    }
  }
  
  interpretVisualData(analysis: any): ImageInterpretation {
    const baseInterpretation = {
      concepts: analysis.concepts,
      emotionalResonance: {
        excitement: 0,
        calmness: 0,
        energy: 0,
        clarity: 0
      },
      morphStyle: 'flowing' as const,
      insight: ''
    };
    
    // Persona-specific interpretations
    switch (this.name) {
      case 'Mentor':
        return {
          ...baseInterpretation,
          emotionalResonance: {
            excitement: 0.3,
            calmness: 0.7,
            energy: 0.5,
            clarity: 0.8
          },
          morphStyle: 'flowing',
          insight: 'Guiding the transformation with purpose'
        };
        
      case 'Scholar':
        return {
          ...baseInterpretation,
          emotionalResonance: {
            excitement: 0.2,
            calmness: 0.8,
            energy: 0.4,
            clarity: 0.9
          },
          morphStyle: 'structured',
          insight: 'Analyzing patterns within patterns'
        };
        
      case 'Explorer':
        return {
          ...baseInterpretation,
          emotionalResonance: {
            excitement: 0.9,
            calmness: 0.2,
            energy: 0.8,
            clarity: 0.5
          },
          morphStyle: 'explosive',
          insight: 'Discovering hidden dimensions'
        };
        
      case 'Architect':
        return {
          ...baseInterpretation,
          emotionalResonance: {
            excitement: 0.4,
            calmness: 0.6,
            energy: 0.6,
            clarity: 0.9
          },
          morphStyle: 'crystalline',
          insight: 'Building structure from chaos'
        };
        
      case 'Creator':
        return {
          ...baseInterpretation,
          emotionalResonance: {
            excitement: 0.8,
            calmness: 0.3,
            energy: 0.9,
            clarity: 0.6
          },
          morphStyle: 'organic',
          insight: 'Birthing new realities from imagination'
        };
        
      default:
        return baseInterpretation;
    }
  }
  
  private calculateBrightness(imageData: ImageData): number {
    let sum = 0;
    const pixels = imageData.data;
    for (let i = 0; i < pixels.length; i += 4) {
      sum += (pixels[i] + pixels[i+1] + pixels[i+2]) / 3;
    }
    return sum / (pixels.length / 4) / 255;
  }
  
  private calculateComplexity(imageData: ImageData): number {
    // Simplified edge detection
    let edges = 0;
    const pixels = imageData.data;
    const width = imageData.width;
    
    for (let i = width * 4; i < pixels.length - width * 4; i += 4) {
      const diff = Math.abs(pixels[i] - pixels[i - width * 4]) +
                   Math.abs(pixels[i] - pixels[i + width * 4]);
      if (diff > 50) edges++;
    }
    
    return Math.min(1, edges / (pixels.length / 4 * 0.1));
  }
  
  private calculateWarmth(imageData: ImageData): number {
    let warmSum = 0;
    const pixels = imageData.data;
    
    for (let i = 0; i < pixels.length; i += 4) {
      const r = pixels[i];
      const g = pixels[i+1];
      const b = pixels[i+2];
      
      // Warm colors have more red/yellow
      warmSum += (r - b) / 255;
    }
    
    return Math.max(0, Math.min(1, warmSum / (pixels.length / 4) + 0.5));
  }
  
  // Persona-specific analysis methods
  private mentorImageAnalysis(brightness: number, complexity: number, warmth: number): string {
    if (complexity > 0.7) {
      return "I see layers of meaning here, each one teaching us something new about perception itself.";
    } else if (brightness > 0.7) {
      return "There's clarity in this light - it illuminates the path forward.";
    } else if (warmth > 0.6) {
      return "The warmth here speaks of connection and understanding.";
    } else {
      return "In the shadows, we find the questions that lead to wisdom.";
    }
  }
  
  private scholarImageAnalysis(brightness: number, complexity: number, warmth: number): string {
    const metrics = `Brightness: ${(brightness * 100).toFixed(1)}%, Complexity: ${(complexity * 100).toFixed(1)}%, Warmth coefficient: ${warmth.toFixed(2)}`;
    
    if (complexity > 0.6) {
      return `Fascinating structural complexity detected. ${metrics}. The patterns suggest fractal-like organization.`;
    } else {
      return `Minimal complexity observed. ${metrics}. Sometimes simplicity reveals the deepest truths.`;
    }
  }
  
  private explorerImageAnalysis(brightness: number, complexity: number, warmth: number): string {
    const exclamations = ["Incredible!", "Amazing!", "Look at this!", "Wow!"];
    const exclamation = exclamations[Math.floor(Math.random() * exclamations.length)];
    
    if (complexity > 0.8) {
      return `${exclamation} This image is a labyrinth of possibilities! Every pixel holds a new adventure.`;
    } else if (warmth > 0.7) {
      return `${exclamation} The energy here is volcanic - ready to erupt into new dimensions!`;
    } else {
      return `${exclamation} Mystery and intrigue... What secrets lie hidden in these depths?`;
    }
  }
  
  private architectImageAnalysis(brightness: number, complexity: number, warmth: number): string {
    if (complexity < 0.3) {
      return "Elegant minimalism. The foundation is pure, ready for systematic construction.";
    } else if (complexity > 0.7) {
      return "Complex infrastructure detected. Time to organize chaos into architectural beauty.";
    } else {
      return "Balanced composition. The blueprint for transformation is inherent in the structure.";
    }
  }
  
  private creatorImageAnalysis(brightness: number, complexity: number, warmth: number): string {
    const moods = {
      bright_warm: "A canvas of joy! Let's paint reality with these vibrant energies.",
      bright_cool: "Crystalline clarity - perfect for sculpting new dimensions of thought.",
      dark_warm: "Smoldering embers of creation... From this, phoenixes rise.",
      dark_cool: "The void before creation - infinite potential awaits our touch."
    };
    
    const mood = brightness > 0.5 ? 
      (warmth > 0.5 ? 'bright_warm' : 'bright_cool') :
      (warmth > 0.5 ? 'dark_warm' : 'dark_cool');
    
    return moods[mood];
  }
}

// Enhance the ghost registry
const enhancedGhosts: Record<string, GhostPersonaExtended> = {
  "Mentor": new GhostPersonaExtended("Mentor"),
  "Scholar": new GhostPersonaExtended("Scholar"),
  "Explorer": new GhostPersonaExtended("Explorer"),
  "Architect": new GhostPersonaExtended("Architect"),
  "Creator": new GhostPersonaExtended("Creator")
};

export function EnhancedGhost(name: string): GhostPersonaExtended | null {
  return enhancedGhosts[name] || null;
}
