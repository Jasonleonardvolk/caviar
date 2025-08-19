// Maps device tilt to a virtual head offset. Tune gains in the ctor.
export class DeviceOrientationProvider {
    constructor(gainX = 0.02, // px->normalized: degrees * gainX -> ~0..1 space
    gainY = 0.015) {
        this.gainX = gainX;
        this.gainY = gainY;
        this.name = 'DeviceOrientation';
        this.onPose = null;
        this.running = false;
    }
    async isSupported() {
        return typeof window !== 'undefined' && 'DeviceOrientationEvent' in window;
    }
    async ensurePermission() {
        // iOS needs a user gesture + permission. We can only *request* here; UI must prompt.
        const anyDOE = DeviceOrientationEvent;
        if (typeof anyDOE?.requestPermission === 'function') {
            try {
                const res = await anyDOE.requestPermission();
                if (res !== 'granted')
                    throw new Error('DeviceOrientation permission not granted');
            }
            catch {
                // swallow; we'll still try and let the handler no-op
            }
        }
    }
    async start(onPose) {
        this.onPose = onPose;
        this.running = true;
        await this.ensurePermission();
        this.handler = (e) => {
            if (!this.running)
                return;
            const t = performance.now();
            // beta: [-180..180] front/back tilt  | gamma: [-90..90] left/right tilt
            const beta = (e.beta ?? 0); // degrees
            const gamma = (e.gamma ?? 0);
            // Map tilt to virtual lateral/vertical head offset in normalized screen space
            const nx = gamma * this.gainX;
            const ny = beta * this.gainY;
            // Orientation as Euler (rad). alpha often unreliable indoors; keep 0.
            const yaw = 0;
            const pitch = (beta * Math.PI) / 180;
            const roll = (gamma * Math.PI) / 180;
            const pose = { p: [nx, ny, 0], r: [yaw, pitch, roll], confidence: 0.85, t };
            this.onPose?.(pose);
        };
        window.addEventListener('deviceorientation', this.handler);
    }
    stop() {
        this.running = false;
        if (this.handler)
            window.removeEventListener('deviceorientation', this.handler);
        this.handler = undefined;
        this.onPose = null;
    }
}
