let alive=false, raf=0, ctx:CanvasRenderingContext2D;
const COL = '#00d0ff'; const N=48; const cols:number[]=[]; const ys:number[]=[];

export default {
  start(canvas: HTMLCanvasElement){
    alive=true; ctx = canvas.getContext('2d')!;
    const w = canvas.width = canvas.clientWidth, h = canvas.height = canvas.clientHeight;
    const cw = Math.floor(w/20); for(let i=0;i<cw;i++){ cols[i]=i*20; ys[i]=Math.random()*h; }
    const glyphs = '░▒▓◆◇●○✦✧✩✪✫✬✭✮✯☼☾☽✶✷✸✹✺';
    const frame=()=>{
      if(!alive) return;
      canvas.width = canvas.clientWidth; canvas.height = canvas.clientHeight;
      ctx.globalCompositeOperation='source-over'; ctx.fillStyle='rgba(0,0,0,0.12)'; ctx.fillRect(0,0,canvas.width,canvas.height);
      ctx.globalCompositeOperation='lighter'; ctx.fillStyle=COL; ctx.font='20px ui-monospace';
      for(let i=0;i<cols.length;i++){
        const ch = glyphs[Math.floor(Math.random()*glyphs.length)];
        ctx.fillText(ch, cols[i], ys[i]);
        ys[i] = (ys[i]+20 + Math.random()*8) % canvas.height;
      }
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
  },
  stop(){ alive=false; cancelAnimationFrame(raf); },
  onResize(){},
  onBoost(on:boolean){ /* tie to stronger glow via layer duplication if needed */ }
} satisfies import('../ShowController').ShowAPI;
