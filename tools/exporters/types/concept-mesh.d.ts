export interface ConceptNode {
  id: string;
  label?: string;
  // Optional position from your mesh; if absent, we will grid-place.
  pos?: { x: number; y: number; z?: number };
}

export interface ConceptEdge {
  source: string;
  target: string;
  weight?: number;
}

export interface ConceptMesh {
  nodes: ConceptNode[];
  edges?: ConceptEdge[];
  meta?: Record<string, unknown>;
}