// Type definitions for conceptHologramRenderer.js

export interface ConceptHologramRenderer {
  init(config?: any): Promise<void>;
  render(concepts: any[]): void;
  update(deltaTime: number): void;
  addConcept(concept: any): void;
  removeConcept(conceptId: string): void;
  clear(): void;
  getRenderer(): any;
  [key: string]: any;
}

declare const conceptHologramRenderer: ConceptHologramRenderer;
export default conceptHologramRenderer;
