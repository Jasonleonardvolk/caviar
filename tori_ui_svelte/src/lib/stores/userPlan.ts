import plans from '$lib/../../static/plans.json';
export type PlanId = keyof typeof plans;
let current: PlanId = (typeof window !== 'undefined'
  && (localStorage.getItem('iris.plan') as PlanId)) || 'free';

export const getPlan = () => plans[current];
export const setPlan = (p: PlanId) => {
  current = p;
  if (typeof window !== 'undefined') localStorage.setItem('iris.plan', p);
};
export const canExport = (fmt: string) => getPlan().export.includes(fmt);
export const maxDuration = () => getPlan().maxDurationSec ?? 10;
export const needsWatermark = () => !!getPlan().watermark;
export const getStripePriceId = () => (getPlan() as any).stripePriceId as string | undefined;