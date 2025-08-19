const DEFAULT_KEY = 'ht_prompt_v1';
const STYLE = `
:host { all: initial; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
.wrap { position: fixed; left: 50%; transform: translateX(-50%); bottom: 18px;
        background: rgba(0,0,0,.78); color: #e5e7eb; padding: 10px 12px; border-radius: 999px;
        box-shadow: 0 10px 30px rgba(0,0,0,.35); display: flex; gap: 8px; align-items: center;
        backdrop-filter: blur(6px); -webkit-backdrop-filter: blur(6px); }
.cta { appearance: none; border: 0; padding: 8px 12px; border-radius: 999px; font-weight: 600; cursor: pointer; }
.enable { background: #10b981; color: #02130d; }
.dismiss { background: transparent; color: #9ca3af; }
.hide { animation: fadeout .18s ease forwards; }
@keyframes fadeout { to { opacity: 0; transform: translate(-50%, 10px); } }
.toast { position: fixed; left: 50%; transform: translateX(-50%); bottom: 18px; background: rgba(16,185,129,.95);
         color:#02130d; padding:8px 12px; border-radius:999px; font-weight:600; box-shadow:0 10px 30px rgba(0,0,0,.3);}
`;
function daysAgo(n) {
    return Date.now() - n * 24 * 60 * 60 * 1000;
}
export function mountHeadTrackingPrompt(opts) {
    const storageKey = opts.storageKey ?? DEFAULT_KEY;
    const snoozeDays = opts.snoozeDays ?? 30;
    const val = localStorage.getItem(storageKey);
    if (val) {
        try {
            const state = JSON.parse(val);
            if (state.status === 'accepted')
                return () => { };
            if (state.status === 'dismissed' && state.ts > daysAgo(snoozeDays))
                return () => { };
        }
        catch { /* ignore parse errors */ }
    }
    const root = document.createElement('div');
    const shadow = root.attachShadow({ mode: 'closed' });
    const style = document.createElement('style');
    style.textContent = STYLE;
    const wrap = document.createElement('div');
    wrap.className = 'wrap';
    wrap.innerHTML = `
    <span>Enable head tracking for smoother 3D?</span>
    <button class="cta dismiss" part="dismiss">Not now</button>
    <button class="cta enable"  part="enable">Enable</button>
  `;
    shadow.append(style, wrap);
    let shown = false;
    const show = () => {
        if (shown)
            return;
        document.body.appendChild(root);
        shown = true;
    };
    const hide = () => {
        wrap.classList.add('hide');
        setTimeout(() => { if (root.isConnected)
            root.remove(); }, 200);
    };
    const toast = (msg) => {
        const t = document.createElement('div');
        t.className = 'toast';
        t.textContent = msg;
        document.body.appendChild(t);
        setTimeout(() => t.remove(), 1600);
    };
    const enableBtn = wrap.querySelector('.enable');
    const dismissBtn = wrap.querySelector('.dismiss');
    enableBtn.onclick = async () => {
        try {
            const ok = await opts.onEnable();
            if (ok) {
                localStorage.setItem(storageKey, JSON.stringify({ status: 'accepted', ts: Date.now() }));
                hide();
                toast('Head tracking enabled');
            }
            else {
                toast('Couldn\'t enable. Using mouse.');
            }
        }
        catch {
            toast('Permission declined');
        }
    };
    dismissBtn.onclick = () => {
        localStorage.setItem(storageKey, JSON.stringify({ status: 'dismissed', ts: Date.now() }));
        hide();
        opts.onDismiss?.();
    };
    // Show on first user intent (non-pesky)
    const revealOnce = () => { show(); window.removeEventListener('pointerdown', revealOnce, true); };
    if (opts.autoShow ?? true)
        window.addEventListener('pointerdown', revealOnce, true);
    return () => {
        window.removeEventListener('pointerdown', revealOnce, true);
        if (root.isConnected)
            root.remove();
    };
}
