// pose_controller.ts - deviceorientation -> simple 6DoF-ish camera pose (smoothed)
export type Pose = { tx: number, ty: number, tz: number, rx: number, ry: number, rz: number };

export class PoseController {
  private pose: Pose = { tx: 0, ty: 0, tz: 0.35, rx: 0, ry: 0, rz: 0 };
  private alpha = 0.12; // smoothing

  startDeviceOrientation() {
    const handler = (e: DeviceOrientationEvent) => {
      // Map yaw/pitch to ry/rx; clamp small range
      const ry = ((e.gamma ?? 0) / 30);   // left/right tilt
      const rx = -((e.beta ?? 0) / 30);   // forward/back tilt
      // Low-pass filter
      this.pose.ry = this.pose.ry + this.alpha * (ry - this.pose.ry);
      this.pose.rx = this.pose.rx + this.alpha * (rx - this.pose.rx);
      // Derive small xy translation from rotation
      this.pose.tx = this.pose.ry * 0.05;
      this.pose.ty = this.pose.rx * 0.05;
    };
    if (typeof DeviceOrientationEvent !== 'undefined' && (DeviceOrientationEvent as any).requestPermission) {
      (DeviceOrientationEvent as any).requestPermission().then((s: string) => {
        if (s === 'granted') window.addEventListener('deviceorientation', handler, true);
      }).catch(() => {/* ignore */});
    } else {
      window.addEventListener('deviceorientation', handler, true);
    }
  }

  getPose(): Pose { return this.pose; }
}