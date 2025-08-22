let alive=false, raf=0, gl:WebGL2RenderingContext, prog:WebGLProgram, t=0;

const vs = `#version 300 es
precision highp float; out vec2 uv;
void main(){ vec2 p = vec2((gl_VertexID<<1 & 2) - 1, (gl_VertexID & 2) - 1);
uv = p; gl_Position = vec4(p,0,1); }`;

const fs = `#version 300 es
precision highp float; in vec2 uv; out vec4 o;
uniform float u_t;
float ring(vec2 p, float r, float w){ float d = abs(length(p)-r); return smoothstep(w,0.0,d); }
void main(){
  vec2 p = uv * 1.2;
  float a = 0.0;
  for(int i=0;i<6;i++){
    float R = 0.15 + 0.12*float(i) + 0.02*sin(u_t*0.8 + float(i)*1.7);
    a += ring(p, R, 0.02) * (1.0 - float(i)/6.0);
  }
  vec3 col = vec3(0.15, 0.8, 1.0);
  o = vec4(col * a*1.8, a);
}`;

function compile(gl:WebGL2RenderingContext, src:string, type:number){
  const s = gl.createShader(type)!; gl.shaderSource(s,src); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)!);
  return s;
}
function link(gl:WebGL2RenderingContext, vsSrc:string, fsSrc:string){
  const p = gl.createProgram()!;
  gl.attachShader(p, compile(gl,vsSrc,gl.VERTEX_SHADER));
  gl.attachShader(p, compile(gl,fsSrc,gl.FRAGMENT_SHADER));
  gl.linkProgram(p); if(!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p)!);
  return p;
}

export default {
  start(canvas: HTMLCanvasElement){
    alive=true;
    gl = canvas.getContext('webgl2', { alpha:true, premultipliedAlpha:false })!;
    prog = link(gl, vs, fs);
    const ut = gl.getUniformLocation(prog,'u_t');
    const frame = ()=>{
      if(!alive) return; t += 0.016;
      gl.viewport(0,0,canvas.width=canvas.clientWidth, canvas.height=canvas.clientHeight);
      gl.clearColor(0,0,0,1); gl.clear(gl.COLOR_BUFFER_BIT);
      gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE); // additive
      gl.useProgram(prog); gl.uniform1f(ut, t);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
  },
  stop(){ alive=false; cancelAnimationFrame(raf); },
  onResize(){},
  onBoost(on:boolean){ /* optional: widen rings */ }
} satisfies import('../ShowController').ShowAPI;
