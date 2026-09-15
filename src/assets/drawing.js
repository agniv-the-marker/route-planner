/* Canvas input; route interaction is shared with the text page. */
(() => {
  'use strict';
  const NS = 'http://www.w3.org/2000/svg';
  const state = { strokes: [], undo: [], revision: 0, request: null, maps: [], urls: [] };
  const $ = (selector, root = document) => root?.querySelector(selector);
  const clone = value => JSON.parse(JSON.stringify(value));
  const clamp = value => Math.max(0, Math.min(1, value));
  let canvas, ctx, active;

  function status(message = '', error = false) {
    const el = $('#status');
    el.textContent = message;
    el.classList.toggle('error', error);
  }
  function persist() {
    try { localStorage.setItem('route-sculptor-drawing', JSON.stringify(state.strokes)); } catch (_) {}
  }
  function disposeResults() {
    state.maps.splice(0).forEach(map => map.destroy?.() ?? map.stop?.());
    state.urls.splice(0).forEach(url => URL.revokeObjectURL(url));
    $('#route-list').replaceChildren();
    $('#results').hidden = true;
  }
  function invalidate() {
    state.revision++;
    state.request?.abort();
    disposeResults();
    status();
    persist();
  }
  function snapshot() {
    state.undo.push(clone(state.strokes));
    if (state.undo.length > 40) state.undo.shift();
  }
  function updateTools() {
    $('[data-action="undo"]').disabled = !state.undo.length;
    $('[data-action="clear"]').disabled = !state.strokes.length;
  }
  function paint() {
    if (!ctx) return;
    ctx.clearRect(0, 0, 511, 511);
    ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--map-paper') || '#f1f2ec';
    ctx.fillRect(0, 0, 511, 511);
    ctx.strokeStyle = '#d75332';
    ctx.lineWidth = 2.2;
    ctx.lineCap = ctx.lineJoin = 'round';
    state.strokes.forEach((stroke, index) => {
      ctx.beginPath();
      stroke.forEach((p, i) => i ? ctx.lineTo(p[0] * 511, p[1] * 511) : ctx.moveTo(p[0] * 511, p[1] * 511));
      ctx.stroke();
      const from = state.strokes[index - 1]?.at(-1), to = stroke[0];
      if (from && to) {
        ctx.save(); ctx.setLineDash([3, 3]); ctx.lineWidth = 1.1;
        ctx.beginPath(); ctx.moveTo(from[0] * 511, from[1] * 511); ctx.lineTo(to[0] * 511, to[1] * 511); ctx.stroke(); ctx.restore();
      }
    });
    updateTools();
  }
  function history(action) {
    if (active) return;
    if (action === 'clear' && state.strokes.length) { snapshot(); state.strokes = []; }
    else if (action === 'undo' && state.undo.length) state.strokes = state.undo.pop();
    else return;
    invalidate(); paint();
  }
  function setupCanvas() {
    canvas = $('#draw-canvas');
    ctx = canvas?.getContext('2d');
    if (!ctx) return status('This browser cannot draw on the canvas.', true);
    const point = event => {
      const rect = canvas.getBoundingClientRect();
      return [clamp((event.clientX - rect.left) / rect.width), clamp((event.clientY - rect.top) / rect.height)];
    };
    canvas.addEventListener('pointerdown', event => {
      if (event.button !== 0 || active) return;
      canvas.setPointerCapture(event.pointerId);
      snapshot(); active = [point(event)]; state.strokes.push(active);
      invalidate(); paint();
    });
    canvas.addEventListener('pointermove', event => {
      if (!active) return;
      const p = point(event), last = active.at(-1);
      if (Math.hypot(p[0] - last[0], p[1] - last[1]) > .002) { active.push(p); paint(); }
    });
    const finish = () => {
      if (!active) return;
      if (active.length < 2) { state.strokes.pop(); state.undo.pop(); }
      active = null; paint(); persist();
    };
    ['pointerup', 'pointercancel', 'lostpointercapture'].forEach(name => canvas.addEventListener(name, finish));
    paint();
  }
  function svgElement(name, attributes) {
    const node = document.createElementNS(NS, name);
    Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value));
    return node;
  }
  function path(points) {
    return points.map((p, index) => `${index ? 'L' : 'M'}${Number(p[0]).toFixed(2)},${Number(p[1]).toFixed(2)}`).join(' ');
  }
  /* Every result card clones the whole map, and the street path alone is ~690 KB.
     Publish one copy the clones can <use>, so the geometry is parsed and cached once. */
  function sharedStreets() {
    if (document.getElementById('shared-streets')) return true;
    const source = $('#map-template .street-map .streets');
    if (!source) return false;
    const holder = svgElement('svg', {width: '0', height: '0', 'aria-hidden': 'true'});
    holder.style.position = 'absolute';
    const defs = svgElement('defs', {});
    const geometry = source.cloneNode(true);
    geometry.setAttribute('id', 'shared-streets');
    geometry.removeAttribute('class');
    defs.append(geometry); holder.append(defs); document.body.append(holder);
    return true;
  }
  function renderMap(route) {
    const map = $('#map-template .street-map').cloneNode(true);
    const svg = $('.map-canvas', map);
    const streets = $('.streets', svg);
    if (streets && sharedStreets()) {
      const reference = svgElement('use', {class: 'streets'});
      reference.setAttribute('href', '#shared-streets');
      reference.setAttributeNS('http://www.w3.org/1999/xlink', 'xlink:href', '#shared-streets');
      streets.replaceWith(reference);
    }
    svg.setAttribute('viewBox', '0 0 511 511');
    svg.setAttribute('aria-label', 'Bicycle route in San Francisco');
    const overlay = svgElement('g', { class: 'result-overlay' });
    const raw = route.target_pixels || route.source_pixels || [];
    const strokes = Array.isArray(raw[0]?.[0]) ? raw : (raw.length ? [raw] : []);
    strokes.forEach(stroke => overlay.append(svgElement('path', { class: 'source-overlay', d: path(stroke) })));
    for (let i = 1; i < strokes.length; i++) {
      const from = strokes[i - 1].at(-1), to = strokes[i][0];
      if (from && to) overlay.append(svgElement('path', { class: 'source-overlay', d: path([from, to]) }));
    }
    overlay.append(svgElement('path', { class: 'route-option chosen', d: path(route.xy_pixels) }));
    svg.append(overlay);
    const toolbar = $('.map-bottom', map);
    const play = document.createElement('button');
    play.type = 'button'; play.className = 'route-play'; play.textContent = '▶';
    play.setAttribute('aria-label', 'Play route');
    toolbar.prepend(play);
    return map;
  }
  function renderResults(data) {
    disposeResults();
    const routes = (Array.isArray(data.routes) ? data.routes : []).filter(route => route.xy_pixels?.length > 1 && route.gpx);
    if (!routes.length) { status('No matching ride found. Try simplifying the drawing.', true); return; }
    const list = $('#route-list');
    routes.forEach((route, index) => {
      const card = document.createElement('article'); card.className = 'route-card';
      const heading = document.createElement('h3');
      heading.textContent = `${(Number(route.distance_m) / 1609.344).toFixed(1)} mi`;
      const wrap = document.createElement('div'); wrap.className = 'route-map';
      const map = renderMap(route); wrap.append(map);
      if (route.profile_html) {
        const profile = document.createElement('div'); profile.className = 'route-elevation';
        profile.innerHTML = route.profile_html; wrap.append(profile);
      }
      const download = document.createElement('a');
      const url = URL.createObjectURL(new Blob([route.gpx], { type: 'application/gpx+xml' }));
      state.urls.push(url); download.className = 'map-action';
      download.href = url; download.download = `route-${index + 1}.gpx`; download.textContent = 'download GPX ↗';
      ($('.map-actions', map) || $('.map-bottom', map)).append(download);
      card.append(heading, wrap); list.append(card);
      const controller = window.RoutePlayback?.bind(wrap);
      if (controller) state.maps.push(controller);
    });
    $('#results').hidden = false;
    status(data.status === 'incomplete' ? 'Search time limit reached; these rides are ready to download.' : '');
  }
  function setBusy(busy) {
    const button = $('#submit');
    button.disabled = busy;
    button.classList.toggle('is-busy', busy);
    button.dataset.busyLabel = busy ? 'finding ride' : '';
    button.setAttribute('aria-label', busy ? 'Finding ride' : 'Find a ride');
    button.setAttribute('aria-busy', String(busy));
    button.textContent = busy ? 'finding ride' : 'find a ride ↗';
  }
  async function submit() {
    if (state.request) return;
    if (!state.strokes.some(stroke => stroke.length > 1)) return status('Draw a line on the canvas first.', true);
    const revision = state.revision, controller = new AbortController();
    state.request = controller;
    let timedOut = false;
    // The server's own search budget is 20 s; leave room for a cold start rather than
    // aborting a search that is still running.
    const timeout = setTimeout(() => { timedOut = true; controller.abort(); }, 90000);
    setBusy(true); status();
    try {
      const response = await fetch('/api/drawing/fit', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: 'canvas', strokes: clone(state.strokes) }), signal: controller.signal
      });
      const data = await response.json();
      if (revision !== state.revision) return;
      if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : data.message || 'Could not find a ride. Please try again.');
      renderResults(data);
    } catch (error) {
      if (revision === state.revision) status(timedOut ? 'Search timed out. Please try again.' : error.message || 'Could not find a ride.', true);
    } finally {
      clearTimeout(timeout); state.request = null; setBusy(false);
    }
  }
  document.addEventListener('DOMContentLoaded', () => {
    try {
      const saved = JSON.parse(localStorage.getItem('route-sculptor-drawing') || 'null');
      const strokes = Array.isArray(saved) ? saved : saved?.canvas;
      if (Array.isArray(strokes) && strokes.every(stroke => Array.isArray(stroke) && stroke.every(p => Array.isArray(p) && p.length === 2 && p.every(Number.isFinite)))) state.strokes = strokes;
    } catch (_) {}
    setupCanvas();
    document.querySelectorAll('[data-action]').forEach(button => button.addEventListener('click', () => history(button.dataset.action)));
    $('#submit').addEventListener('click', submit);
    window.addEventListener('pagehide', () => { state.request?.abort(); disposeResults(); });
  });
})();
