/* frontend/floater/src/host.ts */
export type FloaterConfig = {
  mount: HTMLCanvasElement | HTMLVideoElement;
  dims: [number, number];
  kernel: 'onnx_waveop'|'splitstep_fft'|'biharmonic';
  params?: Record<string, any>;
  policy?: Partial<Policy>;
  onEvent?: (e: FloaterEvent) => void;
};

export type FloaterEvent =
  | { type: 'metrics:frame'; ms: number; fps: number; backend: string; memMB: number }
  | { type: 'net:stats'; kbps: number; rttMs: number }
  | { type: 'donor:drop'; ts: number }
  | { type: 'onnx:ready'; backend: string }
  | { type: 'fft:backend'; kind: 'subgroup'|'baseline' }
  | { type: 'error'; message: string };

export type Policy = {
  donorMode: 'lan'|'off';
  keyframeInterval: number|'auto';
  compressedReturn: boolean;
  roiThreshold: number|'auto';
  maxMsPerFrame: number;
  maxMemMB: number;
};

export type Floater = {
  start(): void;
  stop(): void;
  setDims(dims: [number, number]): void;
  setParams(params: Record<string, any>): void;
  setPolicy(policy: Partial<Policy>): void;
  destroy(): void;
};

export async function createFloater(cfg: FloaterConfig): Promise<Floater> {
  const worker = new Worker(new URL('./renderer.worker.js', import.meta.url), { type: 'module' });
  const port = worker;

  const isCanvas = (el: Element): el is HTMLCanvasElement => el.tagName.toLowerCase() === 'canvas';
  let offscreen: OffscreenCanvas | undefined;

  if (isCanvas(cfg.mount) && 'transferControlToOffscreen' in cfg.mount) {
    offscreen = (cfg.mount as HTMLCanvasElement).transferControlToOffscreen();
  }

  const onEvent = cfg.onEvent ?? (() => {});

  port.onmessage = (ev: MessageEvent<FloaterEvent>) => {
    onEvent(ev.data);
  };

  function start() {
    port.postMessage({
      type: 'start',
      dims: cfg.dims,
      kernel: cfg.kernel,
      params: cfg.params || {},
      policy: cfg.policy || {},
      mountKind: offscreen ? 'canvas' : 'video',
      transfer: offscreen,
    }, offscreen ? [offscreen] : []);
  }

  function stop() {
    port.postMessage({ type: 'stop' });
  }

  function setDims(dims: [number, number]) {
    port.postMessage({ type: 'setDims', dims });
  }

  function setParams(params: Record<string, any>) {
    port.postMessage({ type: 'setParams', params });
  }

  function setPolicy(policy: Partial<Policy>) {
    port.postMessage({ type: 'setPolicy', policy });
  }

  function destroy() {
    try { port.postMessage({ type: 'stop' }); } catch {}
    worker.terminate();
  }

  return { start, stop, setDims, setParams, setPolicy, destroy };
}
