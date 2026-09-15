# Public hosting

The site lives at **https://routesculptor.bike**. Modal runs the Python application at
`https://nyro-robotics--route-sculptor-web.modal.run`; a Cloudflare Worker on the domain
passes every request through to it. Custom domains on Modal itself need its Team plan.

Because the Worker proxies rather than frames, the address bar follows the app: `/draw`,
`/about`, and any link to them work on their own. The old address
`https://agniv.me/route-planner/` is a redirect page in the `agniv.me` repository
(`route-planner/index.html`); it is the only hosting file that has to live over there.

## Cloudflare, once

1. Point `routesculptor.bike` at Cloudflare's nameservers with the registrar, and add the
   zone in the Cloudflare dashboard.
2. From this directory: `npx wrangler login`, then `npx wrangler deploy`.

`wrangler.toml` claims the apex and `www` as custom domains, so Wrangler creates the
proxied DNS records and the certificate itself. Nothing else belongs in the zone.

## Changing the app

Deploy the application as `docs/HOSTING.md` describes; the Worker needs no redeploy
because it only forwards. Redeploy the Worker when `ORIGIN` changes — that is, when the
Modal workspace or app name changes.

## What the Worker sends

`X-Forwarded-Host` and `X-Forwarded-Proto` carry the public name, since the upstream
request has to keep Modal's own host for routing. Gradio builds its root URL from those
headers, and `public_base_url()` in `src/web.py` uses them for the link-preview tags, so
shared links and card previews say `routesculptor.bike`. Redirects that point back at
`modal.run` are rewritten to the public name on the way out.
