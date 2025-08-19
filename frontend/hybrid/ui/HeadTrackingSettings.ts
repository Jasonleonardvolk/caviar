// ${IRIS_ROOT}\frontend\hybrid\ui\HeadTrackingSettings.ts
import { SensorManager, type SensorState } from '../lib/sensors/SensorManager';
import { getPredictorParams, updatePredictorParams } from '../lib/parallaxController';

type MountOpts = { manager: SensorManager };

const STYLE = `
:host { all: initial; font-family: system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }
.kebab { position: fixed; top: calc(env(safe-area-inset-top, 0px) + 12px);
         right: calc(env(safe-area-inset-right, 0px) + 12px);
         width: 44px; height: 44px; border-radius: 12px;
         background: rgba(17,24,39,.92); color: #e5e7eb;
         display:grid; place-items:center; cursor:pointer; user-select:none;
         box-shadow: 0 10px 30px rgba(0,0,0,.35); z-index: 10000; }
.kebab:active { transform: translateY(1px); }
.kebab span { font-size: 20px; line-height: 1; }

.menu { position: fixed; top: calc(env(safe-area-inset-top, 0px) + 60px);
        right: calc(env(safe-area-inset-right, 0px) + 12px);
        min-width: 300px; max-width: 360px;
        background: #0b0b0f; color: #e5e7eb; border-radius: 14px;
        box-shadow: 0 16px 50px rgba(0,0,0,.5); padding: 10px; display:none; z-index: 10001; }
.menu.open { display:block; }

.section { padding: 6px 8px; }
.row { display:flex; align-items:center; justify-content:space-between; gap: 10px; padding: 6px 0; }
hr { border:0; height:1px; background:#111827; margin: 6px 0; }

.toggle { appearance:none; width:46px; height:26px; background:#1f2937; border-radius:999px; position:relative; outline:none; cursor:pointer; }
.toggle:checked { background:#10b981; }
.toggle:before { content:""; position:absolute; top:3px; left:3px; width:20px; height:20px; background:#fff; border-radius:999px; transition:left .15s; }
.toggle:checked:before { left:23px; }

select, input[type=number] { background:#111827; color:#e5e7eb; border:1px solid #1f2937; border-radius:8px; padding:6px; }
select { width: 170px; }
input[type=number]{ width: 84px; }

.controls { display:grid; grid-template-columns: 1fr 1fr; gap: 8px; padding-top: 6px; }
.subtle { color:#9ca3af; font-size:12px; margin-top:-2px; }
.btnrow { display:flex; gap:8px; justify-content:flex-end; padding-top: 8px; }
.btn { background:#111827; color:#e5e7eb; border:1px solid #1f2937; border-radius:8px; padding:6px 10px; cursor:pointer; }
.btn:active { transform: translateY(1px); }

@media (pointer:fine) {
  .kebab:hover { filter: brightness(1.05); }
}
`;

function el<K extends keyof HTMLElementTagNameMap>(tag: K, cls?: string) {
  const n = document.createElement(tag); if (cls) n.className = cls; return n;
}
const isIOS = /iPhone|iPad|iPod/i.test(navigator.userAgent);

export function mountHeadTrackingSettings({ manager }: MountOpts) {
  const root = el('div');
  const shadow = root.attachShadow({ mode: 'closed' });
  const style = el('style'); style.textContent = STYLE;

  // Kebab button (three dots on web/Android, ellipsis on iOS)
  const kebab = el('button','kebab') as HTMLButtonElement;
  kebab.setAttribute('aria-label','More settings');
  kebab.setAttribute('aria-haspopup','menu');
  kebab.innerHTML = `<span>${isIOS ? '...' : '⋮'}</span>`;

  const menu = el('div','menu'); menu.setAttribute('role','menu');

  // === Sections ===
  // Header
  const sec0 = el('div','section');
  const title = el('div'); title.textContent = 'Head Tracking';
  title.style.fontWeight = '700';
  sec0.append(title);

  // Enable toggle
  const sec1 = el('div','section');
  const rowToggle = el('div','row');
  const labToggle = el('div'); labToggle.textContent = 'Enabled';
  const toggle = document.createElement('input'); toggle.type='checkbox'; toggle.className='toggle'; toggle.setAttribute('aria-label','Enable head tracking');
  rowToggle.append(labToggle, toggle);
  const subToggle = el('div','subtle'); subToggle.textContent = 'Uses FaceID/WebXR/tilt when available; falls back to mouse.';
  sec1.append(rowToggle, subToggle);

  // Provider
  const sec2 = el('div','section');
  const rowProv = el('div','row');
  const labProv = el('div'); labProv.textContent = 'Provider';
  const select = document.createElement('select'); select.setAttribute('aria-label','Head tracking provider');
  ['auto','FaceIDDepth(iOS)','WebXR(inline)','DeviceOrientation','Mouse'].forEach(v => {
    const o = document.createElement('option'); o.value = v; o.text = v; select.appendChild(o);
  });
  rowProv.append(labProv, select);
  const subProv = el('div','subtle'); subProv.textContent = 'Auto prefers FaceID/WebXR when supported.';
  sec2.append(rowProv, subProv);

  // Tuning
  const sec3 = el('div','section');
  const h2 = el('div'); h2.textContent = 'Tuning'; h2.style.fontWeight = '600';
  const grid = el('div','controls');
  const mkNum = (label: string, key: keyof ReturnType<typeof getPredictorParams>) => {
    const wrap = el('div');
    const l = el('div'); l.textContent = label; l.className='subtle';
    const i = document.createElement('input'); i.type='number'; i.step='0.01'; (i as any).dataset.key = key; i.title = label;
    wrap.append(l,i); return i;
  };
  const nAlpha = mkNum('alpha','alpha' as any);
  const nBeta  = mkNum('beta','beta' as any);
  const nGamma = mkNum('gamma','gamma' as any);
  const nMCut  = mkNum('minCutoff','euroMinCutoff' as any);
  const nEBeta = mkNum('euroBeta','euroBeta' as any);
  const nDCut  = mkNum('dCutoff','euroDerivCutoff' as any);
  [nAlpha,nBeta,nGamma,nMCut,nEBeta,nDCut].forEach(x => grid.appendChild(x.parentElement!));
  const btns = el('div','btnrow');
  const btnApply = el('button','btn'); btnApply.textContent = 'Apply';
  const btnReset = el('button','btn'); btnReset.textContent = 'Defaults';
  btns.append(btnReset, btnApply);
  sec3.append(h2, grid, btns);

  // Assemble
  menu.append(sec0, sec1, hr(), sec2, hr(), sec3);
  shadow.append(style, kebab, menu);
  document.body.appendChild(root);

  // State sync
  const sync = () => {
    const s: SensorState = manager.getState();
    toggle.checked = s.enabled;
    select.value = s.providerPref;
    const p = getPredictorParams() as any;
    nAlpha.value = (p.alpha ?? 0.85).toString();
    nBeta.value  = (p.beta  ?? 0.45).toString();
    nGamma.value = (p.gamma ?? 0.06).toString();
    nMCut.value  = (p.euroMinCutoff ?? 1.0).toString();
    nEBeta.value = (p.euroBeta ?? 0.25).toString();
    nDCut.value  = (p.euroDerivCutoff ?? 1.0).toString();
  };
  manager.onChange(sync); sync();

  // Interactions
  let open = false;
  const close = () => { open=false; menu.classList.remove('open'); };
  const openMenu = () => { open=true; menu.classList.add('open'); };

  kebab.onclick = (e) => {
    e.stopPropagation();
    open ? close() : openMenu();
  };
  document.addEventListener('click', (e) => {
    // close if clicking outside
    if (!root.contains(e.target as Node)) close();
  }, true);

  // a11y keyboard
  kebab.onkeydown = (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open ? close() : openMenu(); }
    if (e.key === 'Escape') { close(); }
  };

  toggle.onchange = async () => {
    if (toggle.checked) await manager.enable();
    else await manager.disable();
  };
  select.onchange = async () => { await manager.setProviderPref(select.value as any); };

  btnApply.onclick = () => {
    updatePredictorParams({
      alpha: +nAlpha.value, beta: +nBeta.value, gamma: +nGamma.value,
      euroMinCutoff: +nMCut.value, euroBeta: +nEBeta.value, euroDerivCutoff: +nDCut.value,
    });
  };
  btnReset.onclick = () => {
    updatePredictorParams({ alpha:0.85, beta:0.45, gamma:0.06, euroMinCutoff:1.0, euroBeta:0.25, euroDerivCutoff:1.0 });
    sync();
  };

  function hr(){ return document.createElement('hr'); }
  return () => { if (root.isConnected) root.remove(); };
}
