export type IvfFrame = Uint8Array;

export interface QuiltIvfStreamOptions {
  views: number;
  width: number;
  height: number;
  fps?: number;
  onReady?: () => void;
  onFrame?: (viewIndex: number, frame: IvfFrame) => void;
  onError?: (err: Error) => void;
}

export class QuiltIvfStream {
  private _open = false;
  constructor(public opts: QuiltIvfStreamOptions) {}

  async open(): Promise<void> {
    this._open = true;
    this.opts.onReady?.();
  }

  push(viewIndex: number, frame: IvfFrame): void {
    if (!this._open) return;
    this.opts.onFrame?.(viewIndex, frame);
  }

  close(): void {
    this._open = false;
  }
}

export default QuiltIvfStream;
