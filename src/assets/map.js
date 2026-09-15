function bindMap() {
  if (window.RoutePlayback) {
    window.RoutePlayback.bindAll(element);
    return;
  }
  let loader = document.querySelector('script[data-route-playback]');
  if (!loader) {
    loader = document.createElement('script');
    loader.src = '/drawing-assets/playback.js';
    loader.dataset.routePlayback = 'true';
    document.head.appendChild(loader);
  }
  loader.addEventListener('load', () => window.RoutePlayback.bindAll(element), {once: true});
}
bindMap();
watch('value', bindMap);
const routeMapObserver = new MutationObserver(() => {
  if (window.RoutePlayback) requestAnimationFrame(() => window.RoutePlayback.bindAll(element));
});
routeMapObserver.observe(element, {childList: true, subtree: true});
