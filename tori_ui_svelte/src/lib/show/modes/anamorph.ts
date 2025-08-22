let alive=false, raf=0, gl:WebGL2RenderingContext, prog:WebGLProgram;
let ax=0, ay=0; // device tilt

const vs = `#version 300 es
precision highp float; out vec2 uv;
void main(){ vec2 p = vec2((gl_VertexID<<1 & 2) - 1, (gl_VertexID & 2) - 1);
uv = p; gl_Position = vec4(p,0,1); }`;
const fs = `#version 300 es
precision highp float; in vec2 uv; out vec4 o;
uniform vec2 u_tilt;
// Simple shaded bust via signed distance toy (demo illusion)
float sdHead(vec3 p){
  p.z += 0.25; // bring forward
  float s = length(p - vec3(0.0,0.0,0.2)) - 0.45;
  s = min(s, length(p-vec3(0.18,0.15,0.05))-0.18); // cheek
  return s;
}
vec3 shade(vec3 ro, vec3 rd){
  float t=0., h=0.;
  for(int i=0;i<80;i++){ vec3 p=ro+rd*t; h=sdHead(p); if(h<0.001){break;} t+=h*0.5; if(t>3.) break; }
  vec3 p=ro+rd*t; vec2 e=vec2(0.002,0);
  vec3 n=normalize(vec3(sdHead(p+vec3(e.x,0,0))-sdHead(p-vec3(e.x,0,0)),
                        sdHead(p+vec3(0,e.x,0))-sdHead(p-vec3(0,e.x,0)),
                        sdHead(p+vec3(0,0,e.x))-sdHead(p-vec3(0,0,e.x))));
  float diff = max(dot(n, normalize(vec3(0.4,0.6,0.7))), 0.0);
  float rim = pow(1.0 - max(dot(n,-rd),0.0), 2.0);
  return mix(vec3(0.1,0.8,1.0)*diff + vec3(0.0,0.2,0.4), vec3(0.2,0.9,1.0), rim);
}
void main(){
  vec2 p = uv;
  // tilt-based parallax camera
  vec3 ro = vec3(u_tilt.x*0.6, -u_tilt.y*0.6, -1.5);
  vec3 rd = normalize(vec3(p.x*0.9 - u_tilt.x*0.3, p.y*0.9 + u_tilt.y*0.3, 1.7));
  vec3 col = shade(ro, rd);
  float a = 1.0;
  o = vec4(col, a);
}`;

function link(gl:WebGL2RenderingContext, vs:string, fs:string){ const p=gl.createProgram()!;
  const S=(s:string,t:number)=>{const sh=gl.createShader(t)!; gl.shaderSource(sh,s); gl.compileShader(sh);
  if(!gl.getShaderParameter(sh,gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh)!); return sh; };
  gl.attachShader(p,S(vs,gl.VERTEX_SHADER)); gl.attachShader(p,S(fs,gl.FRAGMENT_SHADER));
  gl.linkProgram(p); if(!gl.getProgramParameter(p,gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p)!); return p; }

export default {
  start(canvas: HTMLCanvasElement){
    alive=true;
    gl = canvas.getContext('webgl2', { alpha:true, premultipliedAlpha:false })!;
    prog = link(gl, vs, fs);
    const ut = gl.getUniformLocation(prog,'u_tilt');

    const devTilt = (e:DeviceOrientationEvent)=>{ ax = (e.gamma||0)/45; ay = (e.beta||0)/45; };
    if ('DeviceOrientationEvent' in window) window.addEventListener('deviceorientation', devTilt);

    const frame = ()=>{
      if(!alive) return;
      gl.viewport(0,0,canvas.width=canvas.clientWidth, canvas.height=canvas.clientHeight);
      gl.clearColor(0,0,0,1); gl.clear(gl.COLOR_BUFFER_BIT);
      gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE);
      gl.useProgram(prog); gl.uniform2f(ut, ax, ay);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
  },
  stop(){ alive=false; cancelAnimationFrame(raf); },
  onResize(){},
  onBoost(_) {}
} satisfies import('../ShowController').ShowAPI;
