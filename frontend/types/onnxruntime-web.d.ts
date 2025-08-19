declare module "onnxruntime-web" {
  export class InferenceSession {
    static create(model: string | ArrayBufferLike, options?: any): Promise<InferenceSession>;
    run(feeds: Record<string, any>, fetches?: string[] | Record<string, any>, options?: any): Promise<Record<string, any>>;
  }
  export class Tensor<T = number> {
    constructor(type: string, data: T[] | T, dims: number[]);
    readonly type: string;
    readonly data: T[] | T;
    readonly dims: number[];
  }
}
