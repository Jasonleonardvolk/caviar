export function isWebGPUAvailable() {
    return typeof navigator !== "undefined" && !!navigator.gpu;
}
export function isWASMAvailable() {
    return typeof WebAssembly !== "undefined";
}
export function getDeviceTier() {
    const nav = (typeof navigator !== 'undefined') ? navigator : {};
    const ua = nav.userAgent || "";
    const isMobile = /Android|Mobi|iPhone|iPad|Mobile/i.test(ua);
    const mem = nav.deviceMemory || 4;
    const cores = nav.hardwareConcurrency || 4;
    let tier = 'medium';
    if (!('gpu' in nav))
        tier = 'low';
    if (isMobile)
        tier = (mem >= 6 || cores >= 8) ? 'medium' : 'low';
    else
        tier = (mem >= 16 || cores >= 16) ? 'high' : (mem >= 8 || cores >= 8) ? 'medium' : 'low';
    return tier;
}
