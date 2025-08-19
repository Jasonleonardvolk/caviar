// ${IRIS_ROOT}\frontend\hybrid\lib\parallaxMouse.ts
import { ingestInput } from './parallaxController';
import type { Pose } from './tracking/types';

export function attachMouse(canvas: HTMLCanvasElement) {
  const rect = () => canvas.getBoundingClientRect();
  function toPose(e: MouseEvent): Pose {
    const r = rect();
    const nx = (e.clientX - r.left) / Math.max(1, r.width);
    const ny = (e.clientY - r.top)  / Math.max(1, r.height);
    const t = performance.now();
    return { p:[nx, ny, 0], r:[0,0,0], confidence: 1, t };
  }
  const onMove = (e: MouseEvent) => ingestInput(toPose(e));
  canvas.addEventListener('mousemove', onMove);
  return () => canvas.removeEventListener('mousemove', onMove);
}
