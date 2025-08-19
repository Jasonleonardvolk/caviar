import { FaceIdDepthProvider } from './face/FaceIdDepthProvider.ios';
import { WebXRHeadProvider } from './WebXRHeadProvider';
import { DeviceOrientationProvider } from './DeviceOrientationProvider';
import { MouseProvider } from './MouseProvider';
export async function selectBestSensorProvider(canvas) {
    // Preference order: FaceID depth (iOS) > WebXR (inline) > DeviceOrientation > Mouse
    const candidates = [
        new FaceIdDepthProvider(),
        new WebXRHeadProvider(),
        new DeviceOrientationProvider(),
        new MouseProvider(canvas),
    ];
    for (const c of candidates) {
        try {
            if (await c.isSupported())
                return c;
        }
        catch { /* ignore */ }
    }
    // Fallback (should never hit; MouseProvider.isSupported() is true)
    return new MouseProvider(canvas);
}
