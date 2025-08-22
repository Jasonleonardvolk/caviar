import { writable } from 'svelte/store';

export interface ConceptData {
  id: string;
  label: string;
  weight: number;
  connections: string[];
}

export const conceptList = writable<ConceptData[]>([]);
export const selectedConcept = writable<ConceptData | null>(null);
export const conceptGraph = writable<any>(null);
