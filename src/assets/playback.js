(function (global) {
  'use strict';
  const NS = 'http://www.w3.org/2000/svg', bound = new WeakMap(), live = new Set();
  const clamp = (n, a, b) => Math.max(a, Math.min(b, n));
  const routeIn = svg => [...svg.querySelectorAll('.ride, .route-option.chosen, .route-option')]
    .find(path => (path.getAttribute('d') || '').trim().length > 1) || null;
  const localPoint = (svg, event) => {
    const matrix = svg.getScreenCTM();
    return matrix ? new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse()) : {x: 0, y: 0};
  };
  function nearestProgress(path, target, total) {
    let best = 0, bestD = Infinity;
    const scan = (from, to, steps) => {
      for (let i = 0; i <= steps; i += 1) {
        const length = clamp(from + (to - from) * i / steps, 0, total), p = path.getPointAtLength(length);
        const d = (p.x - target.x) ** 2 + (p.y - target.y) ** 2;
        if (d < bestD) { bestD = d; best = length; }
      }
    };
    scan(0, total, 96);
    let radius = total / 96;
    for (let pass = 0; pass < 3; pass += 1) { bestD = Infinity; scan(best - radius, best + radius, 12); radius /= 6; }
    return total ? best / total : 0;
  }
  function profilePoint(path, x) {
    let low = 0, high = path.getTotalLength();
    for (let i = 0; i < 18; i += 1) {
      const mid = (low + high) / 2;
      if (path.getPointAtLength(mid).x < x) low = mid; else high = mid;
    }
    return path.getPointAtLength((low + high) / 2);
  }
  function profileCoordinates(path) {
    const points = [...(path.getAttribute('d') || '').matchAll(/[ML]([\d.+-]+),([\d.+-]+)/g)]
      .map(match => ({x: Number(match[1]), y: Number(match[2])}));
    return points.length > 1 ? points : null;
  }
  function markerIn(svg) {
    let marker = svg.querySelector('.route-progress-marker');
    svg.querySelectorAll('.route-progress, .route-progress-emoji').forEach(node => node.remove());
    if (marker) return marker;
    marker = document.createElementNS(NS, 'g');
    marker.setAttribute('class', 'route-progress-marker');
    marker.setAttribute('role', 'slider'); marker.setAttribute('aria-label', 'Position along route'); marker.setAttribute('tabindex', '0');
    const hit = document.createElementNS(NS, 'circle'); hit.setAttribute('r', '10'); hit.setAttribute('fill', 'transparent');
    const bike = document.createElementNS(NS, 'text'); bike.setAttribute('class', 'route-progress-bike');
    for (const [key, value] of Object.entries({x:'0', y:'0', 'text-anchor':'middle', 'dominant-baseline':'central', 'font-size':'9'})) bike.setAttribute(key, value);
    bike.textContent = '🚲'; marker.append(hit, bike); svg.appendChild(marker); return marker;
  }
  function bind(root) {
    if (!root) return null;
    const svg = root.querySelector('.map-canvas'); if (!svg) return null;
    const route = routeIn(svg), play = root.querySelector('.route-play');
    // The elevation profile is a sibling of the map, so it can arrive after it.
    const profile = root.querySelector('.elevation-profile') || root.parentElement?.querySelector('.elevation-profile');
    const profileSvg = profile?.querySelector('svg');
    // The map subtree can be observed mid-insertion, before the route, the play button
    // and the profile exist. Comparing only the <svg> node would pin that empty first
    // binding forever — that is what left play/pause dead and the elevation dot missing.
    const prior = bound.get(root);
    if (prior && prior.svg === svg && prior.route === route && prior.play === play
        && prior.profileSvg === profileSvg) return prior.controller;
    if (prior) prior.controller.release();
    // Every listener below is tied to this signal, so a rebind cannot leave duplicates.
    const listeners = new AbortController();
    const on = (target, type, handler, options) =>
      target?.addEventListener(type, handler, {...options, signal: listeners.signal});
    const profilePath = profileSvg?.querySelector('.route-elevation-line, path[fill="none"]');
    const profilePoints = profilePath ? profileCoordinates(profilePath) : null;
    const marker = route ? markerIn(svg) : null;
    let dot = profileSvg?.querySelector('.route-elevation-dot');
    if (profileSvg && !dot) { dot = document.createElementNS(NS, 'circle'); dot.setAttribute('class','route-elevation-dot'); dot.setAttribute('r','3'); profileSvg.appendChild(dot); }
    let view = [0,0,511,511], pan = null, viewFrame = null, panFrame = null, shift = null;
    let box = null, touchAction = '', cursor = '';
    // getBoundingClientRect forces layout, so read the box once per gesture rather than per event.
    const measure = () => (box = svg.getBoundingClientRect());
    const setStyle = (action, pointer) => {
      if (action !== touchAction) svg.style.touchAction = touchAction = action;
      if (pointer !== cursor) svg.style.cursor = cursor = pointer;
    };
    const paintView = () => { viewFrame = null; view[0]=clamp(view[0],0,511-view[2]); view[1]=clamp(view[1],0,511-view[3]); svg.setAttribute('viewBox',view.join(' ')); setStyle(view[2]>=511?'pan-y':'none', view[2]>=511?'default':'grab'); };
    const queueView = () => { if (viewFrame === null) viewFrame=requestAnimationFrame(paintView); };
    // While dragging, move the already-rasterised SVG on the compositor instead of
    // re-tessellating ~48k street segments into a new viewBox on every frame.
    const paintShift = () => { panFrame = null; svg.style.transform = shift ? `translate(${shift[0]}px, ${shift[1]}px)` : ''; };
    const queueShift = () => { if (panFrame === null) panFrame=requestAnimationFrame(paintShift); };
    const commitShift = () => {
      if (panFrame !== null) { cancelAnimationFrame(panFrame); panFrame = null; }
      shift = null; svg.style.transform = ''; svg.classList.remove('is-panning'); paintView();
    };
    const zoom = (factor,x=.5,y=.5) => { const size=clamp(view[2]*factor,85,511); view=[view[0]+(view[2]-size)*x,view[1]+(view[3]-size)*y,size,size]; queueView(); };
    const reset = () => { view=[0,0,511,511]; box=null; queueView(); };
    on(svg, 'wheel', e => { e.preventDefault(); const r=box||measure(); zoom(Math.exp(clamp(e.deltaY,-120,120)*.0015),(e.clientX-r.left)/r.width,(e.clientY-r.top)/r.height); },{passive:false});
    on(svg, 'pointerdown', e => { if(e.button!==0||e.target.closest('.route-progress-marker'))return; measure(); pan=[e.pointerId,e.clientX,e.clientY,...view]; shift=[0,0]; svg.classList.add('is-panning'); svg.setPointerCapture(e.pointerId); });
    on(svg, 'pointermove', e => {
      if(!pan||pan[0]!==e.pointerId)return;
      const r=box||measure();
      view[0]=clamp(pan[3]-(e.clientX-pan[1])*pan[5]/r.width,0,511-view[2]);
      view[1]=clamp(pan[4]-(e.clientY-pan[2])*pan[5]/r.height,0,511-view[3]);
      // Translate by exactly the clamped view delta, so the commit on release is pixel-identical.
      shift=[(pan[3]-view[0])*r.width/view[2],(pan[4]-view[1])*r.height/view[3]];
      queueShift();
    }, {passive:true});
    const endPan=e=>{if(pan?.[0]===e.pointerId){pan=null;box=null;commitShift();}}; ['pointerup','pointercancel','lostpointercapture'].forEach(name=>on(svg,name,endPan));
    on(root.querySelector('[data-map="in"]'),'click',()=>zoom(1/1.3)); on(root.querySelector('[data-map="out"]'),'click',()=>zoom(1.3)); on(root.querySelector('[data-map="reset"]'),'click',reset);
    let frame=null, playing=false, progress=0, started=0;
    const length=route?.getTotalLength()||0, duration=clamp(length*22,7000,26000);
    const paintButton=()=>{if(play){play.textContent=playing?'Ⅱ':'▶';play.setAttribute('aria-label',playing?'Pause route direction':'Play route direction');}};
    const setProgress=value=>{ progress=clamp(value,0,1); if(route&&marker){const p=route.getPointAtLength(length*progress); marker.setAttribute('transform',`translate(${p.x} ${p.y})`); marker.setAttribute('aria-valuenow',String(Math.round(progress*100)));} if(profileSvg&&dot){const width=profileSvg.viewBox.baseVal.width||511;let p={x:width*progress,y:35};if(profilePoints){const raw=progress*(profilePoints.length-1),i=Math.min(profilePoints.length-2,Math.floor(raw)),f=raw-i,a=profilePoints[i],b=profilePoints[i+1];p={x:a.x+(b.x-a.x)*f,y:a.y+(b.y-a.y)*f};}else if(profilePath)p=profilePoint(profilePath,width*progress);dot.setAttribute('cx',p.x);dot.setAttribute('cy',p.y);} };
    const stop=()=>{playing=false;if(frame!==null)cancelAnimationFrame(frame);frame=null;paintButton();};
    const release=()=>{stop();listeners.abort();if(viewFrame!==null)cancelAnimationFrame(viewFrame);if(panFrame!==null)cancelAnimationFrame(panFrame);if(scrubFrame!==null)cancelAnimationFrame(scrubFrame);};
    const tick=now=>{if(!playing)return;setProgress((now-started)/duration);if(progress>=1)stop();else frame=requestAnimationFrame(tick);};
    const toggle=e=>{e?.stopPropagation();if(playing){stop();return;}if(progress>=1)setProgress(0);playing=true;started=performance.now()-progress*duration;paintButton();frame=requestAnimationFrame(tick);}; on(play,'click',toggle);
    let markerPointer=null, scrubFrame=null, scrubEvent=null;
    // nearestProgress costs ~136 getPointAtLength calls; run it at most once per frame.
    const runScrub=()=>{scrubFrame=null;if(scrubEvent)setProgress(nearestProgress(route,localPoint(svg,scrubEvent),length));};
    const scrubMap=e=>{scrubEvent=e;if(scrubFrame===null)scrubFrame=requestAnimationFrame(runScrub);};
    on(marker,'pointerdown',e=>{if(e.button!==0)return;e.preventDefault();e.stopPropagation();stop();markerPointer=e.pointerId;marker.setPointerCapture(e.pointerId);scrubMap(e);}); on(marker,'pointermove',e=>{if(markerPointer===e.pointerId)scrubMap(e);},{passive:true}); on(marker,'keydown',e=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;e.preventDefault();stop();if(e.key==='Home')setProgress(0);else if(e.key==='End')setProgress(1);else setProgress(progress+(e.key==='ArrowRight'?.01:-.01));}); const endMarker=e=>{if(markerPointer===e.pointerId)markerPointer=null;}; ['pointerup','pointercancel','lostpointercapture'].forEach(name=>on(marker,name,endMarker));
    let profilePointer=null, profileMatrix=null;
    // The chart is letterboxed: its viewBox is 511x70 but the box is capped at 100px tall,
    // so the plotted line occupies only the middle ~70% of the element. Mapping the pointer
    // across the element's width made dragging run ~1.4x fast and offset. Convert through
    // the screen matrix instead, which is exact whatever the layout does.
    const scrubProfile=e=>{
      const matrix=profileMatrix||(profileMatrix=profileSvg.getScreenCTM()?.inverse());
      if(!matrix)return;
      const width=profileSvg.viewBox.baseVal.width||511;
      setProgress(new DOMPoint(e.clientX,e.clientY).matrixTransform(matrix).x/width);
    };
    on(profileSvg,'pointerdown',e=>{if(e.button!==0)return;e.preventDefault();stop();profilePointer=e.pointerId;profileSvg.setPointerCapture(e.pointerId);scrubProfile(e);}); on(profileSvg,'pointermove',e=>{if(profilePointer===e.pointerId)scrubProfile(e);},{passive:true}); const endProfile=e=>{if(profilePointer===e.pointerId){profilePointer=null;profileMatrix=null;}}; ['pointerup','pointercancel','lostpointercapture'].forEach(name=>on(profileSvg,name,endProfile));
    const controller={setProgress,toggle,stop,reset,release,svg};
    live.add(controller);
    bound.set(root,{svg,route,play,profileSvg,controller});
    if (route && length > 0) { setProgress(0); paintButton(); }
    paintView(); return controller;
  }
  // Gradio replaces the whole map node on every update; without this the old
  // controller's playback loop keeps running against a detached SVG forever.
  const sweep=()=>{for(const controller of live) if(!controller.svg.isConnected){controller.release();live.delete(controller);}};
  const bindAll=(scope=document)=>{
    sweep();
    const roots = scope.matches?.('.route-map') ? [scope] : scope.querySelectorAll('.route-map');
    if (roots.length) roots.forEach(bind);
    else (scope.matches?.('.street-map') ? [scope] : scope.querySelectorAll('.street-map')).forEach(bind);
  };
  global.RoutePlayback={bind,bindAll};
})(window);
