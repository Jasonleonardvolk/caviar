export class FaceIdDepthProvider {
    constructor() {
        this.name = 'FaceIDDepth(iOS)';
        this.onPose = null;
        this.stream = null;
        this.running = false;
    }
    async isSupported() {
        const ua = navigator.userAgent;
        const isIOS = /iPhone|iPad/i.test(ua);
        return isIOS && 'mediaDevices' in navigator;
    }
    async start(onPose) {
        this.onPose = onPose;
        this.running = true;
        this.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
        const loop = () => {
            if (!this.running)
                return;
            const t = performance.now();
            const pose = { p: [0, 0, 0], r: [0, 0, 0], confidence: 0.8, t };
            this.onPose?.(pose);
            requestAnimationFrame(loop);
        };
        requestAnimationFrame(loop);
    }
    stop() {
        this.running = false;
        this.stream?.getTracks().forEach(t => t.stop());
        this.stream = null;
    }
}
