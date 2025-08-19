export class AdaptiveRenderer {
    constructor(targetFPS = 60, measurementInterval = 10000) {
        this.targetFrameTime = 1000 / targetFPS;
        this.measurementInterval = measurementInterval; // Default 10 seconds
        this.lastMeasurementTime = 0;
        this.measurementTimer = null;
        // Initialize with default settings (these will be tuned in initialize())
        this.settings = {
            shaderPrecision: 'float32',
            viewCount: 16,
            resolutionScale: 1.0
        };
    }
    /**
     * Detect device capabilities and adjust rendering settings accordingly.
     * Should be called once on startup. Returns the selected settings.
     */
    async initialize() {
        const tier = this.detectDeviceTier();
        // If WebGPU is available, check for advanced features (like shader float16 support)
        let adapter = null;
        if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
            try {
                adapter = await navigator.gpu.requestAdapter();
            }
            catch (err) {
                adapter = null;
            }
        }
        // Set baseline settings based on device tier
        this.updateSettingsForTier(tier);
        // Adjust shader precision if float16 is supported and beneficial
        if (adapter && adapter.features) {
            // If the device supports 16-bit shader precision
            if (adapter.features.has('shader-f16')) {
                if (tier === 'low' || tier === 'medium') {
                    // Use 16-bit floats on low/medium tier for performance gain
                    this.settings.shaderPrecision = 'float16';
                }
                else {
                    // On high tier, prefer full precision (float32) for quality
                    this.settings.shaderPrecision = 'float32';
                }
            }
        }
        // Optionally measure initial frame performance to fine-tune settings
        try {
            const frameTime = await this.measureFramePerformance();
            if (frameTime > this.targetFrameTime * 1.2) {
                // If actual frame time is significantly higher (slower) than target, reduce settings
                if (this.settings.viewCount > 8) {
                    // Reduce view count (limit minimum to 8)
                    this.settings.viewCount = Math.max(8, Math.floor(this.settings.viewCount * 0.75));
                }
                else if (this.settings.resolutionScale > 0.5) {
                    // If view count is already low, reduce resolution scale
                    this.settings.resolutionScale *= 0.8;
                    if (this.settings.resolutionScale < 0.5) {
                        this.settings.resolutionScale = 0.5;
                    }
                }
            }
        }
        catch (e) {
            console.warn("Frame performance measurement failed or was skipped:", e);
        }
        console.log(`AdaptiveRenderer: Detected tier "${tier}", settings =`, this.settings);
        // Start runtime scaling if interval is set
        if (this.measurementInterval > 0) {
            this.startRuntimeScaling();
        }
        return this.settings;
    }
    /**
     * Start periodic runtime performance measurement and scaling.
     * Responds to thermal throttling and dynamic GPU load.
     */
    startRuntimeScaling() {
        if (this.measurementTimer !== null) {
            return; // Already running
        }
        console.log(`AdaptiveRenderer: Starting runtime scaling (interval=${this.measurementInterval}ms)`);
        // Use setTimeout for compatibility (setInterval might not exist in all environments)
        const scheduleNext = () => {
            this.measurementTimer = setTimeout(async () => {
                await this.runtimeAdjust();
                if (this.measurementTimer !== null) {
                    scheduleNext();
                }
            }, this.measurementInterval);
        };
        scheduleNext();
    }
    /**
     * Stop runtime scaling measurements.
     */
    stopRuntimeScaling() {
        if (this.measurementTimer !== null) {
            clearTimeout(this.measurementTimer);
            this.measurementTimer = null;
            console.log('AdaptiveRenderer: Stopped runtime scaling');
        }
    }
    /**
     * Perform runtime adjustment based on current performance.
     * Called periodically to handle thermal throttling and load changes.
     */
    async runtimeAdjust() {
        const now = performance.now();
        // Skip if measured too recently
        if (now - this.lastMeasurementTime < this.measurementInterval * 0.9) {
            return;
        }
        this.lastMeasurementTime = now;
        try {
            const frameTime = await this.measureFramePerformance();
            const ratio = frameTime / this.targetFrameTime;
            console.log(`AdaptiveRenderer: Runtime measurement - ${frameTime.toFixed(1)}ms (${(1000 / frameTime).toFixed(0)} FPS)`);
            if (ratio > 1.5) {
                // Severely underperforming - aggressive reduction
                console.log('AdaptiveRenderer: Severe throttling detected, reducing quality');
                if (this.settings.resolutionScale > 0.5) {
                    this.settings.resolutionScale = Math.max(0.5, this.settings.resolutionScale * 0.75);
                }
                else if (this.settings.viewCount > 8) {
                    this.settings.viewCount = Math.max(8, Math.floor(this.settings.viewCount * 0.75));
                }
                // Notify listeners of settings change
                this.notifySettingsChange('throttle');
            }
            else if (ratio > 1.2) {
                // Moderately underperforming - gentle reduction
                console.log('AdaptiveRenderer: Mild throttling detected, minor adjustment');
                if (this.settings.resolutionScale > 0.7) {
                    this.settings.resolutionScale = Math.max(0.7, this.settings.resolutionScale * 0.9);
                }
                this.notifySettingsChange('adjust');
            }
            else if (ratio < 0.7) {
                // Overperforming - can increase quality
                const tier = this.detectDeviceTier();
                if (tier === 'high' && this.settings.resolutionScale < 1.0) {
                    this.settings.resolutionScale = Math.min(1.0, this.settings.resolutionScale * 1.1);
                    console.log('AdaptiveRenderer: Performance headroom, increasing resolution scale');
                    this.notifySettingsChange('boost');
                }
                else if (tier === 'medium' && this.settings.viewCount < 24) {
                    this.settings.viewCount = Math.min(24, this.settings.viewCount + 4);
                    console.log('AdaptiveRenderer: Performance headroom, increasing view count');
                    this.notifySettingsChange('boost');
                }
            }
        }
        catch (e) {
            console.warn('AdaptiveRenderer: Runtime measurement failed:', e);
        }
    }
    /**
     * Notify listeners that settings have changed.
     * Override this method to integrate with your rendering pipeline.
     */
    notifySettingsChange(reason) {
        // Dispatch custom event for the application to handle
        if (typeof window !== 'undefined' && typeof CustomEvent !== 'undefined') {
            window.dispatchEvent(new CustomEvent('adaptive-settings-changed', {
                detail: { settings: this.settings, reason }
            }));
        }
    }
    /**
     * Determine a device performance tier ('low', 'medium', 'high') based on hardware information.
     */
    detectDeviceTier() {
        let tier = 'medium';
        if (typeof navigator === 'undefined') {
            return tier;
        }
        const ua = navigator.userAgent || "";
        const isMobile = /Android|Mobi|iPhone|iPad|Mobile/i.test(ua);
        const deviceMem = navigator.deviceMemory || 4;
        const cores = navigator.hardwareConcurrency || 4;
        if (isMobile) {
            // Mobile devices: generally lower performance
            tier = 'low';
            // High-end tablets/phones heuristic: more cores or memory could bump to medium
            if (deviceMem >= 6 || cores >= 8) {
                tier = 'medium';
            }
        }
        else {
            // Desktop/laptop
            if (deviceMem >= 16 || cores >= 16) {
                tier = 'high';
            }
            else if (deviceMem >= 8 || cores >= 8) {
                tier = 'medium';
            }
            else {
                tier = 'low';
            }
        }
        // If WebGPU not available at all, treat as low (older device or browser)
        if (!('gpu' in navigator)) {
            tier = 'low';
        }
        return tier;
    }
    /**
     * Update the settings based on a given device tier.
     * This sets reasonable defaults for that tier.
     */
    updateSettingsForTier(tier) {
        if (tier === 'high') {
            this.settings.viewCount = 32;
            this.settings.resolutionScale = 1.0;
            this.settings.shaderPrecision = 'float32';
        }
        else if (tier === 'medium') {
            this.settings.viewCount = 16;
            this.settings.resolutionScale = 0.8;
            this.settings.shaderPrecision = 'float32';
        }
        else { // 'low'
            this.settings.viewCount = 8;
            this.settings.resolutionScale = 0.5;
            this.settings.shaderPrecision = 'float32';
        }
    }
    /**
     * Measure the current frame rendering performance (approximate).
     * Uses requestAnimationFrame to estimate frame rate over one second.
     * Returns the average frame time in milliseconds.
     */
    measureFramePerformance() {
        return new Promise((resolve, reject) => {
            if (typeof requestAnimationFrame === 'undefined') {
                // In non-browser environment, skip measurement
                return reject("requestAnimationFrame not available");
            }
            let frameCount = 0;
            const startTime = performance.now();
            const sampleDuration = 1000; // measure for 1 second
            function onFrame(timestamp) {
                frameCount++;
                if (timestamp - startTime < sampleDuration) {
                    requestAnimationFrame(onFrame);
                }
                else {
                    const totalTime = timestamp - startTime;
                    const fps = frameCount * 1000 / totalTime;
                    const avgFrameTime = totalTime / frameCount;
                    resolve(avgFrameTime);
                }
            }
            requestAnimationFrame(onFrame);
        });
    }
}
// Usage example (in an async context):
// const adaptive = new AdaptiveRenderer();
// const settings = await adaptive.initialize();
// // Apply settings: e.g., set shader precision, allocate view buffers of length settings.viewCount, etc.
