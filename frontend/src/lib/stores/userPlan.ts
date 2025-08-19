// userPlan.ts
// User subscription plan management store

import { writable, derived } from 'svelte/store';

export interface PlanFeatures {
  watermark: boolean;
  maxVideoLengthSec: number;
  maxResolution: '1080p' | '4K';
  cloudRender: boolean;
  cloudRenderLimit: number | null; // null = unlimited
  exportARkit: boolean;
  premiumTemplates: boolean;
  supportLevel: 'community' | 'standard' | 'priority';
}

export interface UserPlan {
  id: 'basic' | 'plus' | 'pro';
  name: string;
  tier: 0 | 1 | 2;
  priceMonthly: number;
  priceAnnual: number;
  features: PlanFeatures;
  expiresAt?: Date;
  autoRenew: boolean;
}

// Default free plan
const DEFAULT_PLAN: UserPlan = {
  id: 'basic',
  name: 'Basic',
  tier: 0,
  priceMonthly: 0,
  priceAnnual: 0,
  features: {
    watermark: true,
    maxVideoLengthSec: 10,
    maxResolution: '1080p',
    cloudRender: false,
    cloudRenderLimit: 0,
    exportARkit: false,
    premiumTemplates: false,
    supportLevel: 'community'
  },
  autoRenew: false
};

// Create the store
function createUserPlanStore() {
  const { subscribe, set, update } = writable<UserPlan>(DEFAULT_PLAN);
  
  return {
    subscribe,
    
    // Load plan from localStorage or API
    async load() {
      try {
        // Check localStorage first
        const cached = localStorage.getItem('userPlan');
        if (cached) {
          const plan = JSON.parse(cached);
          set(plan);
        }
        
        // TODO: Fetch from API
        // const response = await fetch('/api/user/plan');
        // const plan = await response.json();
        // set(plan);
        // localStorage.setItem('userPlan', JSON.stringify(plan));
      } catch (error) {
        console.error('Failed to load user plan:', error);
        set(DEFAULT_PLAN);
      }
    },
    
    // Upgrade to a new plan
    async upgrade(planId: 'plus' | 'pro', annual: boolean = false) {
      try {
        // TODO: Process payment via Stripe
        // const response = await fetch('/api/billing/checkout', {
        //   method: 'POST',
        //   body: JSON.stringify({ planId, annual })
        // });
        
        // For now, mock the upgrade
        const plans = {
          plus: {
            id: 'plus' as const,
            name: 'Plus',
            tier: 1 as const,
            priceMonthly: 9.99,
            priceAnnual: 99.99,
            features: {
              watermark: false,
              maxVideoLengthSec: 60,
              maxResolution: '4K' as const,
              cloudRender: true,
              cloudRenderLimit: 20,
              exportARkit: false,
              premiumTemplates: true,
              supportLevel: 'standard' as const
            }
          },
          pro: {
            id: 'pro' as const,
            name: 'Pro',
            tier: 2 as const,
            priceMonthly: 19.99,
            priceAnnual: 199.99,
            features: {
              watermark: false,
              maxVideoLengthSec: 300,
              maxResolution: '4K' as const,
              cloudRender: true,
              cloudRenderLimit: null,
              exportARkit: true,
              premiumTemplates: true,
              supportLevel: 'priority' as const
            }
          }
        };
        
        const newPlan = {
          ...plans[planId],
          expiresAt: new Date(Date.now() + (annual ? 365 : 30) * 24 * 60 * 60 * 1000),
          autoRenew: true
        };
        
        set(newPlan);
        localStorage.setItem('userPlan', JSON.stringify(newPlan));
        
        // Track event
        if (window.psiTelemetry) {
          window.psiTelemetry.track('plan_upgraded', {
            from: 'basic',
            to: planId,
            annual
          });
        }
        
        return true;
      } catch (error) {
        console.error('Upgrade failed:', error);
        return false;
      }
    },
    
    // Cancel subscription
    async cancel() {
      try {
        // TODO: Call cancellation API
        // await fetch('/api/billing/cancel', { method: 'POST' });
        
        update(plan => ({
          ...plan,
          autoRenew: false
        }));
        
        // Track event
        if (window.psiTelemetry) {
          window.psiTelemetry.track('plan_cancelled', {
            plan: get(userPlan).id
          });
        }
        
        return true;
      } catch (error) {
        console.error('Cancellation failed:', error);
        return false;
      }
    },
    
    // Check if a feature is available
    hasFeature(feature: keyof PlanFeatures): boolean {
      const plan = get(userPlan);
      return !!plan.features[feature];
    },
    
    // Get remaining cloud render minutes
    getCloudRenderRemaining(): number | null {
      const plan = get(userPlan);
      if (!plan.features.cloudRender) return 0;
      if (plan.features.cloudRenderLimit === null) return null;
      
      // TODO: Track actual usage
      return plan.features.cloudRenderLimit;
    }
  };
}

export const userPlan = createUserPlanStore();

// Derived stores for common checks
export const isPaid = derived(userPlan, $plan => $plan.tier > 0);
export const isPro = derived(userPlan, $plan => $plan.tier === 2);
export const canExportKit = derived(userPlan, $plan => $plan.features.exportARkit);
export const hasWatermark = derived(userPlan, $plan => $plan.features.watermark);

// Helper to get plan by ID
export function getPlanById(id: string): UserPlan | null {
  const plans = {
    basic: DEFAULT_PLAN,
    plus: {
      id: 'plus' as const,
      name: 'Plus',
      tier: 1 as const,
      priceMonthly: 9.99,
      priceAnnual: 99.99,
      features: {
        watermark: false,
        maxVideoLengthSec: 60,
        maxResolution: '4K' as const,
        cloudRender: true,
        cloudRenderLimit: 20,
        exportARkit: false,
        premiumTemplates: true,
        supportLevel: 'standard' as const
      },
      autoRenew: false
    },
    pro: {
      id: 'pro' as const,
      name: 'Pro',
      tier: 2 as const,
      priceMonthly: 19.99,
      priceAnnual: 199.99,
      features: {
        watermark: false,
        maxVideoLengthSec: 300,
        maxResolution: '4K' as const,
        cloudRender: true,
        cloudRenderLimit: null,
        exportARkit: true,
        premiumTemplates: true,
        supportLevel: 'priority' as const
      },
      autoRenew: false
    }
  };
  
  return plans[id as keyof typeof plans] || null;
}

// Initialize on load
if (typeof window !== 'undefined') {
  userPlan.load();
}

// Add to window for telemetry
declare global {
  interface Window {
    psiTelemetry?: {
      track: (event: string, data: any) => void;
    };
  }
}

function get<T>(store: { subscribe: (fn: (value: T) => void) => () => void }): T {
  let value: T;
  store.subscribe(v => value = v)();
  return value!;
}
