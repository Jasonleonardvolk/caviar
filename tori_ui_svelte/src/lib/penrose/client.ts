// D:\Dev\kha\tori_ui_svelte\src\lib\penrose\client.ts
export interface PenroseSolvePayload {
  scene?: string;
  N?: number;
  wavelength_nm?: number;
  z_mm?: number;
  params?: Record<string, any>;
  field?: number[];
}

export interface PenroseSolveResponse {
  ok: boolean;
  assist: string;
  scene: string;
  N: number;
  ts: number;
  note?: string;
  field?: number[];
}

/**
 * Call Penrose solve endpoint via the proxy
 */
export async function solve(payload: PenroseSolvePayload = {}): Promise<PenroseSolveResponse> {
  const defaultPayload = {
    N: 256,
    scene: 'demo',
    ...payload
  };
  
  const response = await fetch('/api/penrose/solve', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(defaultPayload)
  });
  
  if (!response.ok) {
    throw new Error(`Penrose solve failed: ${response.status} ${response.statusText}`);
  }
  
  return response.json();
}

/**
 * Health check for Penrose service
 */
export async function checkHealth(): Promise<{ ok: boolean; service: string }> {
  const response = await fetch('/api/penrose/health');
  if (!response.ok) {
    throw new Error(`Penrose health check failed: ${response.status}`);
  }
  return response.json();
}

/**
 * Get Penrose API documentation URL
 */
export function getDocsUrl(): string {
  return '/api/penrose/docs';
}