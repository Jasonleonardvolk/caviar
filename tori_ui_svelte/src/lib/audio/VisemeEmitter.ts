/**
 * TypeScript interface for Viseme Emitter
 * This wraps the Python viseme_emitter.py script
 */

export interface Viseme {
  viseme: number;
  start: number;
  end: number;
}

export const PHONEME_TO_VISEME: Record<string, number> = {
  "AA": 1, "AE": 1, "AH": 1, "AO": 1, "AW": 2, "AY": 2,
  "B": 3, "CH": 4, "D": 5, "DH": 5, "EH": 1, "ER": 6, "EY": 1,
  "F": 7, "G": 5, "HH": 8, "IH": 1, "IY": 1, "JH": 4, "K": 5,
  "L": 6, "M": 3, "N": 5, "NG": 5, "OW": 1, "OY": 2, "P": 3,
  "R": 6, "S": 4, "SH": 4, "T": 5, "TH": 7, "UH": 1, "UW": 1,
  "V": 7, "W": 2, "Y": 2, "Z": 4, "ZH": 4
};

export class VisemeEmitter {
  /**
   * Load viseme sequence from JSON file
   */
  static async loadVisemeSequence(jsonPath: string): Promise<Viseme[]> {
    const response = await fetch(jsonPath);
    const data = await response.json();
    return data;
  }

  /**
   * Get current viseme at a given timestamp
   */
  static getVisemeAtTime(visemes: Viseme[], time: number): number {
    for (const viseme of visemes) {
      if (time >= viseme.start && time <= viseme.end) {
        return viseme.viseme;
      }
    }
    return 0; // Default/neutral viseme
  }

  /**
   * Convert phoneme to viseme ID
   */
  static phonemeToViseme(phoneme: string): number {
    return PHONEME_TO_VISEME[phoneme] || 0;
  }
}

// If you need to call the Python script from TypeScript (server-side only):
// You would need to set up an API endpoint that executes the Python script
// and returns the results.
