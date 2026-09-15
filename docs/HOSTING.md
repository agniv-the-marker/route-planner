# Public website hosting

The public entry point is https://routesculptor.bike/. Modal serves the Python/Gradio
application at https://nyro-robotics--route-sculptor-web.modal.run/, and a Cloudflare
Worker on the domain passes requests through to it; `hosting/` holds that Worker and its
setup. Because it proxies instead of framing, `/draw` and `/about` are ordinary
addressable pages. https://agniv.me/route-planner/ redirects to the domain.

## Deploy

From this repository, with the existing SF graph and terrain prepared:

```sh
MODAL_PROFILE=nyro-robotics .venv/bin/modal deploy scripts/modal_app.py
```

The old address is `route-planner/index.html` in the separate agniv.me repository, whose
`gh-pages` branch publishes it; it is a redirect and needs no further changes. Deploy the
Worker from `hosting/` with Wrangler. A private repository cannot be a GitHub Pages
submodule. Do not change repository visibility just to publish anything here.

## Credentials

Deployment uses the locally configured `nyro-robotics` Modal profile. Inside Modal,
the app uses its workspace identity to call `route-sculptor-outlines-v2`.
No Modal token is included in HTML, Git, image environment variables, or the Worker.
Use Modal's local authentication flow for a new machine. Keep credentials outside both
repositories; use secret storage if automated deployment is added later.

## Persistence and spending

`route-sculptor-web-state` is mounted at `/state`. It contains `route_cache/` and
`experiments/`, including the public site's paid-call journal. `MonthlyBudget` creates
an independent $100 inference ledger for each calendar month in America/Los_Angeles,
at `experiments/monthly/YYYY-MM.json`. It automatically starts a new month without
erasing past allocations. Local research budgets remain independent.

Existing cached silhouettes were seeded into the public volume. New calls reserve $1
per silhouette (normally four per prompt) until actual billing is reconciled; this is
a conservative allowance, not an exact price. Drawing and cached generation do not
spend it. CPU serving, storage, and other workspace apps are outside this ledger.

To change the allowance, update `MonthlyBudget(cap_usd=100)` in `scripts/modal_app.py`
and redeploy. That default applies to new months. To adjust an existing month's ledger,
download it from the volume, preserve its allocations, and modify only its cap fields
while the website is stopped. Do not reset or overwrite existing allocations.

Explicit volume commits preserve the reservation and uncertain-dispatch marker before
paid inference, the returned call ID, and completed cached images. Keep one web container:
file locks and the Gradio queue are local, not a distributed concurrency mechanism.
Avoid overlapping deployments during active paid requests. Reconcile interrupted calls
before retrying; never reset their journal to force a retry.

The CPU app scales to zero after 120 idle seconds; cold loads can take time. It serves
only the historical SF map, excluding Peninsula research data. The existing GPU worker
is independently deployed and also scales to zero.

## Verify and roll back

Check `/health`, `/`, `/draw`, and `/about` on the Modal URL, then exercise drawing and
GPX download on routesculptor.bike, watching that the address bar follows each mode. Cached `fish` or `heart` can verify text without
fresh inference. Live novel prompts incur inference charges.

To roll back the server, check out the desired source revision and redeploy without
resetting the persistent volume. To unpublish, delete the Worker and stop only the
`route-sculptor` app in Modal; the separately used GPU worker should remain.

## Deployment verification — 2026-09-14

The public URL and mobile embed loaded in Chromium. Canvas drawing returned a route
with an 855-point GPX download through the public frame and no browser errors.
A fresh `a bold circle` request completed four Modal calls, retained their durable IDs,
and produced two valid contour candidates. One fitted an 18.2-mile loop. The September
ledger uses the $100 cap and reserves $4 for those four calls pending reconciliation.
Two selected `fish` contours had no qualifying route within the existing search limit;
this was reported explicitly, without substituting shapes.

Focused diffusion/web tests passed (13), followed by the monthly-rollover test and an
additional test proving storage failure prevents paid dispatch. OpenCV is now declared
in project dependencies, fixing the initial clean-container startup failure.

## Deployment verification — 2026-09-15 loading animation

Chromium loaded https://routesculptor.bike/ with a cold context at 1200×900 and
390×780. The boot shell's spinning bike was visible 1.5 s and 1.1 s in, with
`gradio-app` hidden behind it and exactly one masthead on screen; the hand-over to the
mounted page landed at 2.6 s and 2.4 s, with the map present, the wordmark in EB
Garamond, no horizontal scroll and no console errors. `/draw` and `/about` still serve
their own pages and the address bar follows each. 134 tests pass. No inference was
dispatched, so the monthly ledger is unchanged.

Hoisting the font `@import` out of the boot document's inline stylesheet took the
served page's first paint off the Google Fonts round trip; the remaining ~1.9 s to
first paint is transfer, not blocking. Measured on the live page: TTFB 1.27 s, HTML
body complete at 2.10 s. That page is 934 KB uncompressed (Cloudflare serves it
`br`-encoded): ~780 KB of it is the SF street path inside Gradio's config, 134 KB the
inline terrain PNG and 14 KB the stylesheet. Cutting that payload is untried.

Redeploying invalidates the image, so the first request afterwards cold-starts a fresh
container. That took about 2.5 minutes here and Cloudflare returned 524 to anything
arriving meanwhile. Warm the Modal URL directly after a deploy before checking the
domain; an ordinary scale-to-zero cold start was about 11 s.
