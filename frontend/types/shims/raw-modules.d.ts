declare module "*?raw" { const src: string; export default src; }
declare module "*.wgsl?raw" { const src: string; export default src; }

declare module "glob";
declare module "ktx-parse";

declare module "@playwright/test" {
  export const test: any;
  export const expect: any;
  const _default: any;
  export default _default;
}
