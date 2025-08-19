// ${IRIS_ROOT}\frontend\lib\state\irisSettings.ts
export type IrisSettings = {
  quality: 'standard' | 'high';
  jbuEnabled: boolean;           // edge-aware upsample
  caEnabled: boolean;            // chromatic aberration
  dofEnabled: boolean;           // light depth-of-field
  mblurEnabled: boolean;         // motion blend
  caStrength: number;            // 0..1
  dofStrength: number;           // 0..1
  mblurStrength: number;         // 0..1
};

const KEY = 'iris_settings_v1';

const DEFAULTS: IrisSettings = {
  quality: 'standard',
  jbuEnabled: true,
  caEnabled: false,
  dofEnabled: false,
  mblurEnabled: false,
  caStrength: 0.12,
  dofStrength: 0.18,
  mblurStrength: 0.15,
};

export function loadSettings(): IrisSettings {
  try { return { ...DEFAULTS, ...JSON.parse(localStorage.getItem(KEY) || '{}') }; }
  catch { return { ...DEFAULTS }; }
}
export function saveSettings(s: Partial<IrisSettings>) {
  const m = { ...loadSettings(), ...s };
  localStorage.setItem(KEY, JSON.stringify(m));
  return m;
}
