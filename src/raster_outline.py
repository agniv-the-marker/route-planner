"""Deterministic conversion of a generated raster into an ``OutlineSpec``.

This module deliberately has no model dependency.  It is the trust boundary between
diffusion output (which is untrusted pixels) and the street search (which only sees a
validated, closed polygon).
"""
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import LinearRing, Polygon

from src.outlines import OutlineSpec


@dataclass
class ExtractionResult:
    outline: OutlineSpec | None = None
    mask: Image.Image | None = None
    preview: Image.Image | None = None
    contour: list = field(default_factory=list)
    reason: str | None = None
    diagnostics: dict = field(default_factory=dict)


def _mask_candidates(gray):
    """Return threshold masks for both possible foreground polarities."""
    _, dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return (dark > 0, ~((dark > 0)))


def _component(mask):
    h, w = mask.shape
    # A connected component touching the edge is background/cropped artwork, not a
    # usable centred silhouette.  Do this before choosing the largest component.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype('uint8'), 8)
    choices = []
    for label in range(1, count):
        x, y, cw, ch, area = stats[label]
        if x == 0 or y == 0 or x + cw >= w or y + ch >= h:
            continue
        box = (int(x), int(y), int(cw), int(ch))
        if _is_framing_component(box, int(area), w, h):
            continue
        choices.append((int(area), label, box))
    if not choices:
        return None, None
    _, label, box = max(choices)
    return labels == label, box


def _is_framing_component(box, area, width, height):
    """Identify an enclosing frame or its enclosed background field.

    A generated image can contain a dark border around a light canvas (or a
    circular badge around a subject).  The inverse threshold then makes the
    canvas itself look like the largest centred silhouette.  Such a component
    is a presentation artifact, not subject geometry, so reject it before
    scoring candidates.  The limits are intentionally conservative: a normal
    subject may occupy much of an image, but should not fill both dimensions
    while also covering nearly half the pixels.
    """
    x, y, cw, ch = box
    footprint = cw / float(width) >= .82 and ch / float(height) >= .82
    occupancy = area / float(width * height)
    # Both a filled canvas and a thin enclosing ring are common logo/frame
    # artifacts.  Their near-canvas footprint is a stronger signal than the
    # threshold polarity; retain ordinary large silhouettes with intermediate
    # occupancy (including the supported white-on-dark case).
    near_canvas_edge = x / float(width) <= .03 and y / float(height) <= .03
    return footprint and (occupancy >= .4 or occupancy <= .06 or near_canvas_edge)


def _add_collinear_vertices(poly, target):
    """Raise vertex count with points on existing edges only."""
    poly = np.asarray(poly, dtype=float)
    while len(poly) < target:
        lengths = np.linalg.norm(np.roll(poly, -1, axis=0) - poly, axis=1)
        edge = int(np.argmax(lengths))
        midpoint = (poly[edge] + poly[(edge + 1) % len(poly)]) / 2
        poly = np.insert(poly, edge + 1, midpoint, axis=0)
    return poly


def _simplify(contour, minimum=8, maximum=48, mask=None):
    perimeter = cv2.arcLength(contour, True)
    # Choose the first epsilon that gives a representable polygon.  Increasing the
    # epsilon is deterministic and preserves sharp silhouette corners.
    for fraction in np.linspace(.001, .05, 80):
        poly = cv2.approxPolyDP(contour, perimeter * fraction, True).reshape(-1, 2)
        if len(poly) < minimum - 1:
            poly = _add_collinear_vertices(poly, minimum - 1)
        if not len(poly) <= maximum - 1:
            continue
        closed = np.vstack((poly, poly[0]))
        if not LinearRing(closed).is_simple or Polygon(closed).area <= 1e-6:
            continue
        if mask is not None:
            raster = np.zeros(mask.shape, dtype='uint8')
            cv2.fillPoly(raster, [np.rint(poly).astype('int32')], 1)
            intersection = np.logical_and(raster, mask).sum()
            union = np.logical_or(raster, mask).sum()
            if not union or intersection / float(union) < .85:
                continue
        return poly
    return None


def extract_outline(image: Image.Image, interpretation='generated silhouette') -> ExtractionResult:
    """Extract one centred filled silhouette, or return an explainable rejection."""
    rgb = image.convert('RGB')
    gray = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    options = []
    for polarity, raw in enumerate(_mask_candidates(gray)):
        # One bounded repair pass: close tiny anti-aliased cracks then fill holes.
        repaired = cv2.morphologyEx(raw.astype('uint8'), cv2.MORPH_CLOSE,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0
        component, box = _component(repaired)
        if component is None:
            continue
        x, y, cw, ch = box
        area = int(component.sum())
        ratio = area / float(w * h)
        if _is_framing_component(box, area, w, h):
            continue
        if not (.015 <= ratio <= .75 and cw >= w * .08 and ch >= h * .08):
            continue
        aspect = max(cw / ch, ch / cw)
        if aspect > 6:
            continue
        # prefer a reasonably sized, centred subject; this resolves polarity without
        # assuming that SD used a white background.
        cx, cy = x + cw / 2, y + ch / 2
        centered = 1 - np.hypot(cx - w / 2, cy - h / 2) / np.hypot(w / 2, h / 2)
        options.append((centered - abs(ratio - .28), polarity, component, box))
    if not options:
        return ExtractionResult(reason='No centred, uncropped silhouette was found.')
    _, polarity, mask, box = max(options, key=lambda x: x[0])
    contours, _ = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return ExtractionResult(reason='The silhouette boundary could not be traced.')
    contour = max(contours, key=cv2.contourArea)
    poly = _simplify(contour, mask=mask)
    if poly is None:
        return ExtractionResult(reason='The silhouette is too detailed to route reliably.')
    # Pixel y already increases downwards, the coordinate convention used by OutlineSpec.
    points = poly.astype(float)
    points[:, 0] /= max(1, w - 1)
    points[:, 1] /= max(1, h - 1)
    points = np.vstack((points, points[0]))
    try:
        spec = OutlineSpec.parse({'interpretation': interpretation[:160], 'points': points.tolist()})
    except ValueError as exc:
        return ExtractionResult(reason=f'Invalid extracted outline: {exc}')
    mask_image = Image.fromarray(np.where(mask, 0, 255).astype('uint8')).convert('RGB')
    preview = rgb.copy()
    draw = ImageDraw.Draw(preview)
    draw.line([tuple(p) for p in poly] + [tuple(poly[0])], fill='#e04b3f', width=max(2, w // 180))
    contour_mask = np.zeros(mask.shape, dtype='uint8')
    cv2.fillPoly(contour_mask, [np.rint(poly).astype('int32')], 1)
    intersection = np.logical_and(contour_mask, mask).sum()
    union = np.logical_or(contour_mask, mask).sum()
    return ExtractionResult(spec, mask_image, preview, points.tolist(), diagnostics={
        'polarity': 'dark-on-light' if polarity == 0 else 'light-on-dark',
        'bbox': box, 'vertices': len(points),
        'occupied_area': round(float(mask.mean()), 5),
        'retained_area': round(float(mask.sum()) / (w * h), 5),
        'contour_iou': round(float(intersection) / float(union), 5) if union else 0.0,
    })
