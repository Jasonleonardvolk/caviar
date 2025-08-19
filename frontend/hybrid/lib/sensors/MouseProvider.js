export class MouseProvider {
    constructor(canvas) {
        this.canvas = canvas;
        this.name = 'Mouse';
        this.onPose = null;
        this.running = false;
    }
    isSupported() { return true; }
    async start(onPose) {
        this.onPose = onPose;
        this.running = true;
        const rect = () => this.canvas.getBoundingClientRect();
        this.handler = (e) => {
            if (!this.running)
                return;
            const r = rect();
            const nx = (e.clientX - r.left) / Math.max(1, r.width);
            const ny = (e.clientY - r.top) / Math.max(1, r.height);
            const t = performance.now();
            const pose = { p: [nx, ny, 0], r: [0, 0, 0], confidence: 1, t };
            this.onPose?.(pose);
        };
        this.canvas.addEventListener('mousemove', this.handler);
    }
    stop() {
        this.running = false;
        if (this.handler)
            this.canvas.removeEventListener('mousemove', this.handler);
        this.handler = undefined;
        this.onPose = null;
    }
}
