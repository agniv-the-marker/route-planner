# Public website hosting

The public entry point is https://agniv.me/route-planner/.
GitHub Pages serves an iframe wrapper; Modal serves the Python/Gradio application at
https://nyro-robotics--route-sculptor-web.modal.run/.
The browser address remains on agniv.me while drawing, text, and about navigation happen
inside the frame. The wrapper uses the app’s bicycle favicon and Route Sculptor title.
A direct custom domain would allow normal address-bar navigation; this iframe does not
synchronize its internal location with the outer page.

## Deploy

From this repository, with the existing SF graph and terrain prepared:

```sh
MODAL_PROFILE=nyro-robotics .venv/bin/modal deploy scripts/modal_app.py
```

The public wrapper is `route-planner/index.html` in the separate agniv.me repository,
whose `gh-pages` branch publishes it. Deploy server changes here; push wrapper changes
there. A private repository cannot be a GitHub Pages submodule. Do not change repository
visibility just to publish this wrapper.

## Credentials

Deployment uses the locally configured `nyro-robotics` Modal profile. Inside Modal,
the app uses its workspace identity to call `route-sculptor-outlines-v2`.
No Modal token is included in HTML, Git, image environment variables, or an iframe URL.
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
GPX download through the public iframe. Cached `fish` or `heart` can verify text without
fresh inference. Live novel prompts incur inference charges.

To roll back the server, check out the desired source revision and redeploy without
resetting the persistent volume. To unpublish, remove the wrapper from agniv.me and stop
only the `route-sculptor` app in Modal; the separately used GPU worker should remain.

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
