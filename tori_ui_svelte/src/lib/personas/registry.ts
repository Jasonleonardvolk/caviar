export const personaRegistry = new Map();
export function getPersonaByName(name: string) { return null; }
export function getAllPersonas() { return []; }
export function getPersonaDefinition(name: string) {
  return personaRegistry.get(name);
}
