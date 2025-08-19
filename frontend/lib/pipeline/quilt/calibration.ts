import type { DisplayCalib } from './types';

export async function loadCalibration(url: string): Promise<DisplayCalib> {
  const r = await fetch(url, { cache: 'no-store' });
  if (!r.ok) throw new Error(`calibration fetch failed: ${r.status}`);
  const j = await r.json();

  // Accept common LKG-style keys; map to our struct.
  const pitch  = Number(j.pitch ?? j.dpi ?? j.serial?.pitch ?? 50);
  const tilt   = Number(j.tilt  ?? j.serial?.tilt  ?? 0.0);
  const center = Number(j.center ?? j.viewCenter ?? 0.0);
  const subp   = Number(j.subp ?? 0);
  const panelW = Number(j.panelW ?? j.screenW ?? (j.config?.width  ?? 3840));
  const panelH = Number(j.panelH ?? j.screenH ?? (j.config?.height ?? 2160));

  return { pitch, tilt, center, subp, panelW, panelH };
}