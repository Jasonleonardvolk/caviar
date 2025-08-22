// Asset modules
declare module '*.glsl' {
  const shader: string;
  export default shader;
}

declare module '*.wgsl' {
  const shader: string;
  export default shader;
}

declare module '*.txt' {
  const content: string;
  export default content;
}

declare module '*.md' {
  const content: string;
  export default content;
}

// App namespace
declare namespace App {
  interface Locals {
    user?: {
      id: string;
      username: string;
    };
  }
  interface PageData {}
  interface Error {}
  interface Platform {}
}
