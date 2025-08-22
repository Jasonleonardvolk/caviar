import { json } from '@sveltejs/kit';
import matrix from '../../../../config/device-matrix.json';
import { parseHardwareModel } from '$lib/device/ua';
import { resolveCaps, resolveTier } from '$lib/device/deviceMatrix';

// Optional: gentle capability probe if hw is unknown (keeps brand-new phones usable)
async function capabilityProbe(): Promise<'GOLD'|'SILVER'|'UNKNOWN'> {
  try {
    // minimal probe: presence of WebGPU + f16 support
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const nav: any = globalThis.navigator;
    if (!nav?.gpu) return 'UNKNOWN';
    const adapter = await nav.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return 'UNKNOWN';
    // heuristic only - we'll default unknowns to SILVER if adapter exists
    return 'SILVER';
  } catch { return 'UNKNOWN'; }
}

export const GET = async ({ request }) => {
  const ua = request.headers.get('user-agent') ?? '';
  const uaStr = ua.toLowerCase();
  const hw = parseHardwareModel(ua) ?? '';
  let tier = 'UNSUPPORTED';
  let caps: any = null;
  let reason = '';

  try {
    if (hw) {
      caps = resolveCaps(hw);
      tier = resolveTier(hw);
    } else {
      // Check if it's an Apple mobile device without known hardware ID
      const isAppleMobile = /\b(iphone|ipad)\b/.test(uaStr);
      
      if (isAppleMobile) {
        // Unknown Apple device - default to SILVER tier
        // This ensures iPhone 17 etc. work on day one
        tier = 'SILVER';
        caps = { maxN: 256, zernikeModes: 12, serverFallback: false };
        reason = 'ua-apple-unknown-fallback';
      } else {
        // Non-Apple device - try capability probe
        const probe = await capabilityProbe();
        if (probe !== 'UNKNOWN') {
          tier = 'SILVER';
          caps = { maxN: 256, zernikeModes: 12, serverFallback: false };
          reason = 'capability-probe';
        } else {
          reason = 'no-hw-and-no-capability';
        }
      }
    }
  } catch (e: any) {
    reason = e?.message || 'resolve-failed';
  }

  return json({
    ok: tier !== 'UNSUPPORTED',
    tier,
    caps,
    hw,
    ua,
    minSupportedModel: (matrix as any).minSupportedModel,
    reason
  });
};