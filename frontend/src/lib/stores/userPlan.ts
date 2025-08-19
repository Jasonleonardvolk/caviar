import { writable, derived } from 'svelte/store';

export type PlanId = 'free' | 'plus' | 'pro';

type Limits = {
  maxMs: number;
  watermark: boolean;
  arExport: boolean;
  cloudRender: boolean;
};

const DEFAULT_PLAN: PlanId = (typeof localStorage !== 'undefined' &&
  (localStorage.getItem('caviar.plan') as PlanId)) || 'free';

export const plan = writable<PlanId>(DEFAULT_PLAN);
plan.subscribe((p) => {
  try { localStorage.setItem('caviar.plan', p); } catch {}
});

export const limitsByPlan: Record<PlanId, Limits> = {
  free: { maxMs: 10_000,  watermark: true,  arExport: false, cloudRender: false },
  plus: { maxMs: 60_000,  watermark: false, arExport: true,  cloudRender: false },
  pro:  { maxMs: 3_600_000, watermark: false, arExport: true,  cloudRender: true }
};

export const limits = derived(plan, ($p) => limitsByPlan[$p]);

export function requireFeature(feature: keyof Limits, $plan: PlanId): boolean {
  const l = limitsByPlan[$plan];
  return Boolean(l?.[feature]);
}