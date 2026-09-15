() => {
  const query = new URLSearchParams(location.search);
  const enabled = {debug: query.get('debug') === 'true', dev: query.get('dev') === 'true'};
  for (const name of ['debug', 'dev']) document.documentElement.dataset[name] = String(enabled[name]);
  const backdrop = document.getElementById('popup-backdrop');
  let current = null, previous = null;
  const content = ['prompt-bar', 'status', 'route-map', 'results', 'credits'];
  function close() {
    if (!current) return;
    current.classList.remove('is-open');
    current.setAttribute('aria-hidden', 'true');
    backdrop.hidden = true;
    document.body.classList.remove('popup-open');
    content.forEach(id => document.getElementById(id).inert = false);
    current = null;
    previous?.focus();
  }
  function open(name, trigger) {
    if (!enabled[name]) return;
    close();
    previous = trigger;
    current = document.getElementById(`${name}-popup`);
    current.classList.add('is-open');
    current.setAttribute('aria-hidden', 'false');
    backdrop.hidden = false;
    document.body.classList.add('popup-open');
    content.forEach(id => document.getElementById(id).inert = true);
    current.querySelector('[data-close]').focus();
  }
  for (const name of ['debug', 'dev']) {
    const popup = document.getElementById(`${name}-popup`);
    popup.setAttribute('role', 'dialog');
    popup.setAttribute('aria-modal', 'true');
    popup.setAttribute('aria-labelledby', `${name}-title`);
    popup.setAttribute('aria-hidden', 'true');
    popup.querySelector('[data-close]').onclick = close;
    document.querySelector(`[data-open="${name}"]`).onclick = e => open(name, e.currentTarget);
  }
  backdrop.onclick = close;
  // Proxy only the action. Never move nodes owned by Gradio into map HTML.
  document.addEventListener('click', event => {
    const action = event.target.closest('[data-main-action="another"]');
    if (!action) return;
    // The alternative search runs on the server for a second or more. Show the busy
    // bike on the button that was actually clicked, before the round trip starts,
    // and ignore repeat clicks until the map comes back.
    if (action.classList.contains('is-busy')) return;
    window.RouteSculptorBusy.mark(action, 'finding another route');
    document.getElementById('another-button')?.click();
  });
  document.addEventListener('keydown', e => {
    if (!current) return;
    if (e.key === 'Escape') { e.preventDefault(); close(); }
    if (e.key === 'Tab') {
      const items = [...current.querySelectorAll('button, a, input, select, textarea, [tabindex="0"]')]
        .filter(el => !el.disabled && el.getClientRects().length);
      const first = items[0], last = items[items.length - 1];
      if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
    }
  });
}
