// ${IRIS_ROOT}\frontend\hybrid\lib\sensors\DeviceOrientationProvider.ts
import type { ISensorProvider } from './ISensorProvider';
import type { Pose } from '../tracking/types';

// Maps device tilt to a virtual head offset. Tune gains in the ctor.
export class DeviceOrientationProvider implements ISensorProvider {
  name = 'DeviceOrientation';
  private onPose: ((pose: Pose) => void) | null = null;
  private handler?: (e: DeviceOrientationEvent) => void;
  private running = false;

  constructor(
    private gainX = 0.02,  // px->normalized: degrees * gainX -> ~0..1 space
    private gainY = 0.015
  ) {}

  async isSupported(): Promise<boolean> {
    return typeof window !== 'undefined' && 'DeviceOrientationEvent' in window;
  }

  private async ensurePermission(): Promise<void> {
    // iOS needs a user gesture + permission. We can only *request* here; UI must prompt.
    const anyDOE = DeviceOrientationEvent as any;
    if (typeof anyDOE?.requestPermission === 'function') {
      try {
        const res = await anyDOE.requestPermission();
        if (res !== 'granted') throw new Error('DeviceOrientation permission not granted');
      } catch {
        // swallow; we'll still try and let the handler no-op
      }
    }
  }

  async start(onPose: (pose: Pose) => void): Promise<void> {
    this.onPose = onPose;
    this.running = true;
    await this.ensurePermission();

    this.handler = (e: DeviceOrientationEvent) => {
      if (!this.running) return;
      const t = performance.now();
      // beta: [-180..180] front/back tilt  | gamma: [-90..90] left/right tilt
      const beta = (e.beta ?? 0);  // degrees
      const gamma = (e.gamma ?? 0);
      // Map tilt to virtual lateral/vertical head offset in normalized screen space
      const nx = gamma * this.gainX;
      const ny = beta  * this.gainY;

      // Orientation as Euler (rad). alpha often unreliable indoors; keep 0.
      const yaw   = 0;
      const pitch = (beta  * Math.PI) / 180;
      const roll  = (gamma * Math.PI) / 180;

      const pose: Pose = { p: [nx, ny, 0], r: [yaw, pitch, roll], confidence: 0.85, t };
      this.onPose?.(pose);
    };
    window.addEventListener('deviceorientation', this.handler);
  }

  stop(): void {
    this.running = false;
    if (this.handler) window.removeEventListener('deviceorientation', this.handler);
    this.handler = undefined;
    this.onPose = null;
  }
}
