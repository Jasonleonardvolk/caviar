// ${IRIS_ROOT}\frontend\hybrid\lib\sensors\face\FaceIdDepthProvider.ios.ts
import type { ISensorProvider } from '../ISensorProvider';
import type { Pose } from '../../tracking/types';

export class FaceIdDepthProvider implements ISensorProvider {
  name = 'FaceIDDepth(iOS)';
  private onPose: ((pose: Pose) => void) | null = null;
  private stream: MediaStream | null = null;
  private running = false;

  async isSupported(): Promise<boolean> {
    const ua = navigator.userAgent;
    const isIOS = /iPhone|iPad/i.test(ua);
    return isIOS && 'mediaDevices' in navigator;
  }

  async start(onPose: (pose: Pose) => void): Promise<void> {
    this.onPose = onPose; this.running = true;
    this.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' as any }, audio: false });
    const loop = () => {
      if (!this.running) return;
      const t = performance.now();
      const pose: Pose = { p:[0,0,0], r:[0,0,0], confidence: 0.8, t };
      this.onPose?.(pose);
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  stop(): void {
    this.running = false;
    this.stream?.getTracks().forEach(t => t.stop());
    this.stream = null;
  }
}
