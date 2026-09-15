from PIL import Image, ImageDraw

from src.diffusion import candidate_seeds
from src.raster_outline import extract_outline


def star(foreground, background):
    image = Image.new('RGB', (256, 256), background)
    ImageDraw.Draw(image).polygon([(128, 22), (154, 96), (234, 96), (169, 143),
                                   (194, 225), (128, 177), (62, 225), (87, 143),
                                   (22, 96), (102, 96)], fill=foreground)
    return image


def test_extracts_dark_and_light_silhouettes():
    for foreground, background in [('black', 'white'), ('white', 'black')]:
        result = extract_outline(star(foreground, background), 'star')
        assert result.reason is None
        assert 8 <= len(result.outline.points) <= 48
        assert result.outline.interpretation == 'star'


def test_rejects_cropped_and_fragmented_images():
    cropped = Image.new('RGB', (128, 128), 'white')
    ImageDraw.Draw(cropped).rectangle((0, 20, 80, 100), fill='black')
    assert extract_outline(cropped).outline is None
    fragments = Image.new('RGB', (128, 128), 'white')
    draw = ImageDraw.Draw(fragments)
    for x in range(10, 120, 25):
        draw.rectangle((x, 60, x + 5, 65), fill='black')
    assert extract_outline(fragments).outline is None


def test_ignores_inset_frame_and_extracts_foreground():
    image = Image.new('RGB', (256, 256), 'white')
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((4, 4, 251, 251), radius=24, outline='black', width=14)
    draw.polygon([(57, 130), (82, 105), (105, 86), (154, 96), (194, 103),
                  (219, 130), (194, 157), (154, 164), (105, 174),
                  (82, 155)], fill='black')

    result = extract_outline(image, 'fish')

    assert result.reason is None
    assert result.diagnostics['polarity'] == 'dark-on-light'
    x, y, width, height = result.diagnostics['bbox']
    assert width < 180 and height < 120
    assert result.diagnostics['occupied_area'] < .25


def test_rejects_logo_field_instead_of_extracting_invalid_detail():
    # A badge/background without a subject must not become a routeable outline.
    image = Image.new('RGB', (256, 256), '#66605f')
    ImageDraw.Draw(image).ellipse((7, 7, 249, 249), fill='#5f805f', outline='#ddf278', width=3)

    result = extract_outline(image, 'fish')

    assert result.outline is None
    assert result.reason is not None


def test_candidate_seeds_are_deterministic_and_distinct():
    assert candidate_seeds('request-1') == candidate_seeds('request-1')
    assert len(set(candidate_seeds('request-1'))) == 4
    assert candidate_seeds('request-1') != candidate_seeds('request-2')
