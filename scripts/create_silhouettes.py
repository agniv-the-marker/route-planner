#!/usr/bin/env python3
"""Create clean programmatic silhouettes for the concept library.

These are simple 2D filled black shapes on white backgrounds — much cleaner
than SD 1.5 can generate, and guaranteed to have good contour extraction.

Usage:
    PYTHONPATH=. python scripts/create_silhouettes.py
"""

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


OUTPUT_DIR = Path("data/silhouettes")


def draw_star(size=512, points=5, outer_r=0.4, inner_r=0.18):
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2
    pts = []
    for i in range(points * 2):
        angle = math.radians(i * 180 / points - 90)
        r = outer_r * size if i % 2 == 0 else inner_r * size
        pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    draw.polygon(pts, fill=(0, 0, 0))
    return img


def draw_heart(size=512):
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2 + 20
    t = np.linspace(0, 2 * np.pi, 500)
    scale = size * 0.025
    x = scale * 16 * np.sin(t) ** 3 + cx
    y = -scale * (13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)) + cy
    draw.polygon(list(zip(x.tolist(), y.tolist())), fill=(0, 0, 0))
    return img


def draw_circle(size=512):
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    m = int(size * 0.1)
    draw.ellipse([m, m, size - m, size - m], fill=(0, 0, 0))
    return img


def draw_triangle(size=512):
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2
    r = size * 0.4
    pts = []
    for i in range(3):
        angle = math.radians(i * 120 - 90)
        pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    draw.polygon(pts, fill=(0, 0, 0))
    return img


def draw_arrow(size=512):
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    s = size
    # Arrow pointing right
    pts = [
        (0.15*s, 0.4*s), (0.6*s, 0.4*s), (0.6*s, 0.25*s),
        (0.85*s, 0.5*s),
        (0.6*s, 0.75*s), (0.6*s, 0.6*s), (0.15*s, 0.6*s),
    ]
    draw.polygon(pts, fill=(0, 0, 0))
    return img


def draw_lightning(size=512):
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    s = size
    pts = [
        (0.55*s, 0.05*s), (0.3*s, 0.45*s), (0.48*s, 0.45*s),
        (0.35*s, 0.95*s), (0.7*s, 0.4*s), (0.52*s, 0.4*s),
        (0.65*s, 0.05*s),
    ]
    draw.polygon(pts, fill=(0, 0, 0))
    return img


def draw_moon(size=512):
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Crescent: large circle minus offset circle
    m = int(size * 0.1)
    draw.ellipse([m, m, size - m, size - m], fill=(0, 0, 0))
    offset = int(size * 0.2)
    draw.ellipse([m + offset, m - int(size*0.05), size - m + offset, size - m - int(size*0.05)], fill=(255, 255, 255))
    return img


def draw_cross(size=512):
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    s = size
    w = 0.15  # arm width as fraction
    draw.rectangle([s*(0.5-w), s*0.1, s*(0.5+w), s*0.9], fill=(0, 0, 0))
    draw.rectangle([s*0.1, s*(0.5-w), s*0.9, s*(0.5+w)], fill=(0, 0, 0))
    return img


def draw_diamond(size=512):
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2
    rx, ry = size * 0.35, size * 0.42
    pts = [(cx, cy - ry), (cx + rx, cy), (cx, cy + ry), (cx - rx, cy)]
    draw.polygon(pts, fill=(0, 0, 0))
    return img


def draw_horse(size=512):
    """Detailed horse silhouette (side view, facing left)."""
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    s = size / 512
    points = [
        (55*s, 105*s), (48*s, 100*s), (40*s, 92*s), (38*s, 82*s),
        (42*s, 72*s), (50*s, 62*s), (62*s, 54*s), (76*s, 48*s),
        (88*s, 44*s), (86*s, 26*s), (96*s, 36*s),
        (104*s, 32*s), (112*s, 22*s), (114*s, 40*s),
        (120*s, 50*s), (132*s, 62*s), (148*s, 82*s),
        (168*s, 108*s), (188*s, 138*s), (205*s, 162*s),
        (218*s, 172*s), (228*s, 168*s), (238*s, 166*s),
        (260*s, 170*s), (285*s, 172*s), (310*s, 170*s),
        (335*s, 168*s), (355*s, 172*s),
        (370*s, 180*s), (380*s, 192*s),
        (390*s, 190*s), (410*s, 178*s), (430*s, 175*s),
        (448*s, 182*s), (460*s, 200*s), (465*s, 225*s),
        (460*s, 255*s), (448*s, 278*s), (432*s, 290*s),
        (418*s, 288*s), (410*s, 275*s), (405*s, 255*s),
        (398*s, 232*s), (392*s, 212*s),
        (385*s, 220*s), (380*s, 240*s), (375*s, 262*s),
        (370*s, 280*s), (365*s, 295*s),
        (368*s, 315*s), (372*s, 340*s), (375*s, 365*s),
        (374*s, 390*s), (370*s, 415*s), (368*s, 438*s),
        (366*s, 452*s),
        (376*s, 458*s), (382*s, 455*s), (384*s, 448*s),
        (382*s, 425*s), (378*s, 400*s), (376*s, 378*s),
        (378*s, 358*s), (380*s, 340*s),
        (375*s, 325*s), (365*s, 312*s),
        (355*s, 320*s), (348*s, 342*s), (342*s, 365*s),
        (338*s, 390*s), (334*s, 415*s), (332*s, 438*s),
        (330*s, 452*s),
        (340*s, 458*s), (348*s, 455*s), (350*s, 448*s),
        (348*s, 425*s), (345*s, 402*s), (340*s, 378*s),
        (335*s, 355*s), (328*s, 335*s), (318*s, 318*s),
        (300*s, 308*s), (275*s, 312*s), (250*s, 314*s),
        (225*s, 312*s), (205*s, 308*s),
        (200*s, 315*s), (198*s, 335*s), (195*s, 358*s),
        (192*s, 382*s), (188*s, 408*s), (186*s, 432*s),
        (184*s, 452*s),
        (194*s, 458*s), (202*s, 455*s), (204*s, 448*s),
        (202*s, 428*s), (200*s, 405*s), (202*s, 382*s),
        (206*s, 358*s), (210*s, 338*s), (215*s, 320*s),
        (210*s, 310*s), (200*s, 305*s),
        (188*s, 310*s), (180*s, 330*s), (174*s, 355*s),
        (168*s, 380*s), (163*s, 408*s), (160*s, 432*s),
        (158*s, 452*s),
        (168*s, 458*s), (176*s, 455*s), (178*s, 448*s),
        (176*s, 428*s), (175*s, 405*s), (178*s, 382*s),
        (182*s, 358*s), (185*s, 335*s), (188*s, 315*s),
        (182*s, 298*s), (170*s, 278*s), (158*s, 258*s),
        (145*s, 238*s), (130*s, 215*s), (115*s, 192*s),
        (100*s, 168*s), (88*s, 148*s), (78*s, 130*s),
        (70*s, 118*s), (62*s, 112*s),
    ]
    draw.polygon(points, fill=(0, 0, 0))
    return img


def draw_cat(size=512):
    """Simple cat silhouette (sitting, side view)."""
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    s = size / 512
    # Body
    draw.ellipse([150*s, 220*s, 370*s, 440*s], fill=(0, 0, 0))
    # Head
    draw.ellipse([180*s, 120*s, 340*s, 260*s], fill=(0, 0, 0))
    # Left ear
    draw.polygon([(195*s, 145*s), (175*s, 70*s), (235*s, 130*s)], fill=(0, 0, 0))
    # Right ear
    draw.polygon([(305*s, 145*s), (335*s, 70*s), (275*s, 130*s)], fill=(0, 0, 0))
    # Tail
    t = np.linspace(0, np.pi * 1.2, 50)
    tail_x = 350*s + 80*s * np.sin(t)
    tail_y = 380*s - 120*s * (1 - np.cos(t)) / 2
    for i in range(len(t) - 1):
        draw.line([(tail_x[i], tail_y[i]), (tail_x[i+1], tail_y[i+1])],
                  fill=(0, 0, 0), width=int(20*s))
    # Front paws
    draw.rectangle([190*s, 400*s, 230*s, 460*s], fill=(0, 0, 0))
    draw.rectangle([280*s, 400*s, 320*s, 460*s], fill=(0, 0, 0))
    return img


def draw_fish(size=512):
    """Simple fish silhouette."""
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    s = size / 512
    # Body ellipse
    draw.ellipse([80*s, 170*s, 380*s, 340*s], fill=(0, 0, 0))
    # Tail
    draw.polygon([(350*s, 255*s), (450*s, 170*s), (450*s, 340*s)], fill=(0, 0, 0))
    # Eye (white circle)
    draw.ellipse([140*s, 230*s, 175*s, 265*s], fill=(255, 255, 255))
    return img


def draw_bird(size=512):
    """Simple flying bird silhouette."""
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    s = size / 512
    # Body
    draw.ellipse([180*s, 230*s, 350*s, 310*s], fill=(0, 0, 0))
    # Head
    draw.ellipse([310*s, 210*s, 380*s, 270*s], fill=(0, 0, 0))
    # Beak
    draw.polygon([(375*s, 235*s), (420*s, 240*s), (375*s, 250*s)], fill=(0, 0, 0))
    # Left wing (up)
    draw.polygon([(200*s, 240*s), (120*s, 120*s), (280*s, 230*s)], fill=(0, 0, 0))
    # Right wing (down)
    draw.polygon([(280*s, 280*s), (200*s, 380*s), (320*s, 290*s)], fill=(0, 0, 0))
    # Tail
    draw.polygon([(180*s, 260*s), (100*s, 240*s), (100*s, 300*s)], fill=(0, 0, 0))
    return img


def draw_tree(size=512):
    """Simple tree silhouette."""
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    s = size / 512
    # Trunk
    draw.rectangle([230*s, 300*s, 280*s, 470*s], fill=(0, 0, 0))
    # Crown (three overlapping circles)
    draw.ellipse([140*s, 80*s, 320*s, 260*s], fill=(0, 0, 0))
    draw.ellipse([190*s, 50*s, 370*s, 230*s], fill=(0, 0, 0))
    draw.ellipse([100*s, 120*s, 280*s, 300*s], fill=(0, 0, 0))
    draw.ellipse([230*s, 100*s, 410*s, 280*s], fill=(0, 0, 0))
    return img


def draw_dog(size=512):
    """Simple dog silhouette (side view)."""
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    s = size / 512
    # Body
    draw.ellipse([130*s, 200*s, 380*s, 360*s], fill=(0, 0, 0))
    # Head
    draw.ellipse([60*s, 140*s, 200*s, 260*s], fill=(0, 0, 0))
    # Snout
    draw.ellipse([40*s, 180*s, 130*s, 240*s], fill=(0, 0, 0))
    # Ear
    draw.ellipse([100*s, 100*s, 170*s, 180*s], fill=(0, 0, 0))
    # Eye (white)
    draw.ellipse([110*s, 170*s, 130*s, 190*s], fill=(255, 255, 255))
    # Front legs
    draw.rectangle([160*s, 330*s, 200*s, 450*s], fill=(0, 0, 0))
    draw.rectangle([220*s, 330*s, 260*s, 450*s], fill=(0, 0, 0))
    # Back legs
    draw.rectangle([300*s, 320*s, 340*s, 450*s], fill=(0, 0, 0))
    draw.rectangle([350*s, 330*s, 390*s, 450*s], fill=(0, 0, 0))
    # Tail
    draw.polygon([(370*s, 220*s), (440*s, 150*s), (430*s, 180*s), (390*s, 230*s)], fill=(0, 0, 0))
    return img


SHAPES = {
    "star": draw_star,
    "heart": draw_heart,
    "circle": draw_circle,
    "triangle": draw_triangle,
    "arrow": draw_arrow,
    "lightning": draw_lightning,
    "moon": draw_moon,
    "cross": draw_cross,
    "diamond": draw_diamond,
    "horse": draw_horse,
    "cat": draw_cat,
    "fish": draw_fish,
    "bird": draw_bird,
    "tree": draw_tree,
    "dog": draw_dog,
}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, draw_fn in SHAPES.items():
        img = draw_fn()
        path = OUTPUT_DIR / f"{name}.png"
        img.save(path)
        print(f"  Created {path}")

    print(f"\n{len(SHAPES)} silhouettes saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
