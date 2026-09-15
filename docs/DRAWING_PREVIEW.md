# Interactive drawing preview

Implemented 2026-09-13 with two GPT-5.6 Luna subagents and parent integration/verification. The existing text handler and historical experiment artifacts remain in place.

Open http://127.0.0.1:7860/draw. The main website header also links to drawing mode.

To start the local server again:

```bash
GRADIO_ANALYTICS_ENABLED=False MODAL_PROFILE=nyro-robotics .venv/bin/python -m uvicorn src.web:create_site --factory --host 127.0.0.1 --port 7860
```

On map: draw a stroke at the desired location, then select **best fit**. Pan is an explicit tool; wheel/buttons zoom. Undo/redo/clear and browser-local saved strokes are supported.

On canvas: draw a shape, then select **fit to map**. The source stays editable and candidate routes appear below it. Each map displays the blue route and dashed placed source, with GPX download. Changing the drawing, mode or distance invalidates previous downloads. Transfers between strokes are visible and included in the routed/exported traversal.

This prototype searches the historical SF graph through the separate StrokeRouter. It does not change the frozen research runner settings. Interactive matching uses one anchor candidate and beam width one, up to 24 placement proposals, 600,000 work units and a 20-second search ceiling. Canvas spans are 1,200/2,250/3,750 m at every maximum-distance setting. Rotations remain ±15° with no mirroring. Larger distance caps still permit smaller artwork. The first fit loads the graph and can take approximately 15 additional seconds; subsequent requests reuse it.

The search is heuristic. Incomplete searches retain completed candidate routes and show their status explicitly. Human recognition has not been established. Turn restrictions and live closures are absent from this graph. Region selection, Peninsula placement, protected-feature editing and partial-edge anchors remain future work. No text-model improvement or paid GPU job was performed here; the existing text handler still uses its prior diffusion implementation.

Verification: 102 tests passed. Chromium exercised both real HTTP fitting flows, downloaded and parsed GPX, checked zoomed drawing coordinates, undo/redo, persisted strokes, stale-response suppression, distance-change invalidation and 390px mobile layout. No JavaScript errors occurred. A square drawn at a fixed location returned one 6.0 km candidate; canvas placement returned three candidates in 7.7 seconds. These are functional examples, not recognition benchmark passes.

Screenshots, a downloaded test GPX and verification metadata are preserved under `outputs/interactive-drawing-20260913/`.
