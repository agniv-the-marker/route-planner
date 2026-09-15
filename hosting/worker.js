// routesculptor.bike -> the Modal app, unchanged.
//
// Modal serves custom domains only on its Team plan, so this Worker sits on the domain
// and passes every request through to the app's modal.run address. The browser stays on
// routesculptor.bike, so /draw and /about are ordinary addressable pages rather than
// states hidden inside an iframe.

const ORIGIN = 'nyro-robotics--route-sculptor-web.modal.run';

export default {
  async fetch(request) {
    const address = new URL(request.url);
    // Copying the request keeps the method, body, and any WebSocket upgrade intact.
    const upstream = new Request(
      `https://${ORIGIN}${address.pathname}${address.search}`, request);
    // Modal routes on Host, so the public name travels in the forwarded headers instead.
    // Gradio and the link-preview tags read these to build absolute URLs.
    upstream.headers.set('X-Forwarded-Host', address.host);
    upstream.headers.set('X-Forwarded-Proto', 'https');

    const response = await fetch(upstream);
    if (!response.headers.has('location')) return response;

    // A redirect the app issues to itself should land back on this domain.
    const moved = new Response(response.body, response);
    const target = new URL(response.headers.get('location'), `https://${ORIGIN}`);
    if (target.host === ORIGIN) {
      target.protocol = 'https:';
      target.host = address.host;
      moved.headers.set('location', target.toString());
    }
    return moved;
  },
};
