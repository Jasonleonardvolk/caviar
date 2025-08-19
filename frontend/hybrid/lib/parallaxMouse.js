// ${IRIS_ROOT}\frontend\hybrid\lib\parallaxMouse.ts
import { ingestInput } from './parallaxController';
export function attachMouse(canvas) {
    const rect = () => canvas.getBoundingClientRect();
    function toPose(e) {
        const r = rect();
        const nx = (e.clientX - r.left) / Math.max(1, r.width);
        const ny = (e.clientY - r.top) / Math.max(1, r.height);
        const t = performance.now();
        return { p: [nx, ny, 0], r: [0, 0, 0], confidence: 1, t };
    }
    const onMove = (e) => ingestInput(toPose(e));
    canvas.addEventListener('mousemove', onMove);
    return () => canvas.removeEventListener('mousemove', onMove);
}
