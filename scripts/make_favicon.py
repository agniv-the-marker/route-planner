"""Render the browser icons from the bicycle drawing.

The drawing is a tall printed mark on paper; pasted straight into a tab it becomes a
pale smudge. This crops to the ink, fits it into a square of paper, and re-inks the
downscaled strokes so they stay solid at 16-48 px.
"""
from pathlib import Path

from PIL import Image, ImageOps

ASSETS = Path(__file__).resolve().parent.parent / 'src' / 'assets'
INK = (35, 71, 133)
PAPER = (247, 245, 239)
# Grey levels that count as fully paper and fully ink in the scan.
PAPER_LEVEL, INK_LEVEL = 215, 110


def _mark() -> Image.Image:
    """The drawing's ink coverage, cropped to the strokes, as a mask."""
    grey = ImageOps.grayscale(Image.open(ASSETS / 'bike-mark.webp').convert('RGB'))
    coverage = grey.point(
        lambda v: max(0, min(255, round((PAPER_LEVEL - v) * 255 / (PAPER_LEVEL - INK_LEVEL)))))
    return coverage.crop(grey.point(lambda v: 255 if v < 170 else 0).getbbox())


def icon(mark: Image.Image, size: int, pad: float, gamma: float) -> Image.Image:
    inner = round(size * (1 - 2 * pad))
    fitted = ImageOps.contain(mark, (inner, inner), Image.LANCZOS)
    # Averaging thin strokes with paper washes them out; the gamma darkens what survives.
    fitted = fitted.point(lambda v: round(255 * (v / 255) ** gamma))
    square = Image.new('RGB', (size, size), PAPER)
    square.paste(Image.new('RGB', fitted.size, INK),
                 ((size - fitted.width) // 2, (size - fitted.height) // 2), fitted)
    return square


def main() -> None:
    mark = _mark()
    # The smallest size loses the frame entirely, so it is drawn larger and harder.
    # 256 is the largest an .ico holds, and Gradio scales the app's installed icons
    # from it, so it is worth carrying.
    sizes = {16: (0.03, 0.45), 32: (0.06, 0.62), 48: (0.06, 0.62), 64: (0.06, 0.7),
             128: (0.06, 0.78), 256: (0.06, 0.8)}
    icons = [icon(mark, size, *settings) for size, settings in sizes.items()]
    # Pillow drops requested sizes larger than the image it saves from, so the largest
    # icon is the base and the hand-tuned smaller ones are matched from append_images.
    icons[-1].save(ASSETS / 'favicon.ico', sizes=[(s, s) for s in sizes],
                   append_images=icons[:-1])
    icon(mark, 180, 0.06, 0.8).save(ASSETS / 'apple-touch-icon.png')
    print(f"wrote {ASSETS / 'favicon.ico'} and {ASSETS / 'apple-touch-icon.png'}")


if __name__ == '__main__':
    main()
