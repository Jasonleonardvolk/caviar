export function showToast(msg: string) {
  console.warn('[Toast]', msg); // replace with your UI toast impl
}

export function handleRenderError(error: Error) {
  showToast(`Rendering error: ${error.message}. Switching to base mode.`);
  window.dispatchEvent(new CustomEvent('switchToBaseRender'));
  fetch('/api/v2/hybrid/log_error', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ error: error.message })
  }).catch(()=>{});
}
