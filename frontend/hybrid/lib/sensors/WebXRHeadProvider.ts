// ${IRIS_ROOT}\frontend\hybrid\lib\sensors\WebXRHeadProvider.ts
/// <reference types="webxr" />

import type { ISensorProvider } from './ISensorProvider';
import type { Pose } from '../tracking/types';

export class WebXRHeadProvider implements ISensorProvider {
  name = 'WebXR(inline)';
  private onPose: ((pose: Pose) => void) | null = null;
  private session: XRSession | null = null;
  private refSpace: XRReferenceSpace | null = null;
  private running = false;

  async isSupported(): Promise<boolean> {
    const anyNav: any = navigator;
    if (!anyNav?.xr?.isSessionSupported) return false;
    try { return await anyNav.xr.isSessionSupported('inline'); }
    catch { return false; }
  }

  async start(onPose: (pose: Pose) => void): Promise<void> {
    this.onPose = onPose; this.running = true;
    const anyNav: any = navigator;
    this.session = await anyNav.xr.requestSession('inline', { requiredFeatures: ['local'] });
    this.refSpace = await this.session.requestReferenceSpace('local');

    const onXRFrame = (_time: DOMHighResTimeStamp, frame: XRFrame) => {
      if (!this.running || !this.session || !this.refSpace) return;
      const pose = frame.getViewerPose(this.refSpace);
      if (pose) {
        const p = pose.transform.position; // meters
        const r = pose.transform.orientation; // quaternion
        // Map quaternion to yaw/pitch/roll (approx); keep simple here
        const ysqr = r.y * r.y;
        // yaw (Z), pitch (Y), roll (X) from quaternion (basic extraction)
        const t0 = 2 * (r.w * r.z + r.x * r.y);
        const t1 = 1 - 2 * (ysqr + r.z * r.z);
        const yaw = Math.atan2(t0, t1);
        const t2 = Math.max(-1, Math.min(1, 2 * (r.w * r.y - r.z * r.x)));
        const pitch = Math.asin(t2);
        const t3 = 2 * (r.w * r.x + r.y * r.z);
        const t4 = 1 - 2 * (r.x * r.x + ysqr);
        const roll = Math.atan2(t3, t4);

        const now = performance.now();
        const poseOut: Pose = { p: [p.x, p.y, p.z], r: [yaw, pitch, roll], confidence: 0.9, t: now };
        this.onPose?.(poseOut);
      }
      this.session.requestAnimationFrame(onXRFrame);
    };
    this.session.requestAnimationFrame(onXRFrame);
  }

  stop(): void {
    this.running = false;
    this.refSpace = null;
    if (this.session) { this.session.end(); this.session = null; }
  }
}
