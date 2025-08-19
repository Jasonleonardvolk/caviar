// ${IRIS_ROOT}\frontend\hybrid\lib\sensors\MouseProvider.ts
import type { ISensorProvider } from './ISensorProvider';
import type { Pose } from '../tracking/types';

export class MouseProvider implements ISensorProvider {
  name = 'Mouse';
  private onPose: ((pose: Pose) => void) | null = null;
  private handler?: (e: MouseEvent) => void;
  private running = false;

  constructor(private canvas: HTMLCanvasElement) {}

  isSupported(): boolean { return true; }

  async start(onPose: (pose: Pose) => void): Promise<void> {
    this.onPose = onPose; this.running = true;
    const rect = () => this.canvas.getBoundingClientRect();
    this.handler = (e: MouseEvent) => {
      if (!this.running) return;
      const r = rect();
      const nx = (e.clientX - r.left) / Math.max(1, r.width);
      const ny = (e.clientY - r.top)  / Math.max(1, r.height);
      const t = performance.now();
      const pose: Pose = { p:[nx, ny, 0], r:[0,0,0], confidence: 1, t };
      this.onPose?.(pose);
    };
    this.canvas.addEventListener('mousemove', this.handler);
  }

  stop(): void {
    this.running = false;
    if (this.handler) this.canvas.removeEventListener('mousemove', this.handler);
    this.handler = undefined;
    this.onPose = null;
  }
}
