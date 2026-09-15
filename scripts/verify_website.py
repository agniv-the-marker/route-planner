"""Exercise the live website in Chromium and save evidence (may invoke inference)."""
import argparse
import json
from pathlib import Path

import gpxpy
from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', default='fish')
    parser.add_argument('--drawing', action='store_true')
    parser.add_argument('--output', default='outputs/website-repair-20260914')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=['--no-sandbox'])
        page = browser.new_page(viewport={'width': 1200, 'height': 1150}, accept_downloads=True)
        page.on('pageerror', lambda error: errors.append(str(error)))
        if args.drawing:
            page.goto('http://127.0.0.1:7860/draw', wait_until='domcontentloaded')
            page.locator('[data-action="clear"]').click()
            box = page.locator('#draw-canvas').bounding_box()
            points = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8), (0.2, 0.2)]
            for index, (x, y) in enumerate(points):
                page.mouse.move(box['x'] + box['width'] * x, box['y'] + box['height'] * y, steps=12)
                if index == 0:
                    page.mouse.down()
            page.mouse.up()
            with page.expect_response('**/api/drawing/fit', timeout=60000) as response:
                page.locator('#submit').click()
            result = response.value.json()
            page.locator('.route-card').first.wait_for()
            cards = page.locator('.route-card')
            assert 'selected' in cards.first.get_attribute('class')
            cards.nth(1).locator('h3').click()
            assert 'selected' in cards.nth(1).get_attribute('class')
            svg = cards.nth(1).locator('.map-canvas')
            initial = svg.get_attribute('viewBox')
            cards.nth(1).locator('[data-map="in"]').click()
            zoomed = svg.get_attribute('viewBox')
            assert zoomed != initial
            svg.scroll_into_view_if_needed()
            bounds = svg.bounding_box()
            page.mouse.move(bounds['x'] + bounds['width'] / 2, bounds['y'] + bounds['height'] / 2)
            page.mouse.down()
            page.mouse.move(bounds['x'] + bounds['width'] / 2 + 70, bounds['y'] + bounds['height'] / 2 + 40, steps=8)
            page.mouse.up()
            assert svg.get_attribute('viewBox') != zoomed
            cards.nth(1).locator('[data-map="reset"]').click()
            assert svg.get_attribute('viewBox') == initial
            svg.focus()
            svg.press('+')
            assert svg.get_attribute('viewBox') != initial
            svg.press('0')
            assert svg.get_attribute('viewBox') == initial
            svg.hover()
            page.mouse.wheel(0, -300)
            page.wait_for_function("document.querySelectorAll('.route-card .map-canvas')[1].getAttribute('viewBox') !== " + json.dumps(initial))
            cards.nth(1).screenshot(path=str(output / 'drawing-interaction.png'))
            page.locator('.route-card').first.screenshot(path=str(output / 'drawing-square.png'))
            with page.expect_download() as download:
                page.locator('.route-card a[download]').first.click()
            path = output / 'drawing-square.gpx'
            download.value.save_as(str(path))
            points = gpxpy.parse(path.read_text()).tracks[0].segments[0].points
            assert len(points) > 2 and not errors
            assert all(37.69 < q.latitude < 37.82 and -122.53 < q.longitude < -122.35 for q in points)
            view = page.locator('.route-card .map-canvas').first.get_attribute('viewBox')
            assert view != '0 0 511 511'
            (output / 'drawing-square.json').write_text(json.dumps({'status': result['status'], 'seconds': result['seconds'], 'routes': len(result['routes']), 'viewBox': view, 'errors': errors, 'gpx_points': len(points)}, indent=2))
            print(result['message'], result['seconds'], flush=True)
            browser.close()
            return
        page.goto('http://127.0.0.1:7860/', wait_until='domcontentloaded')
        page.locator('#prompt-box textarea').fill(args.prompt)
        page.locator('#generate-button').click()
        page.wait_for_function("document.querySelector('#status')?.textContent.includes('of ') && document.querySelector('#status')?.textContent.includes('silhouettes') || document.querySelector('#status')?.textContent.includes('failed:')", timeout=600000)
        images = page.locator('#outline-choices img')
        images.nth(1 if images.count() > 1 else 0).click()
        page.get_by_text('fit selected outline ↗', exact=True).click()
        page.wait_for_function("document.querySelector('#status')?.textContent.includes('Generated route') || document.querySelector('#status')?.textContent.includes('No qualifying')", timeout=180000)
        status = page.locator('#status').inner_text()
        print(status, flush=True)
        page.screenshot(path=str(output / f'text-{args.prompt}.png'), full_page=True)
        evidence = {'prompt': args.prompt, 'status': status, 'errors': errors,
                    'map': page.locator('.map-canvas').evaluate('e => ({viewBox: e.getAttribute("viewBox"), roads: e.querySelector(".streets").getAttribute("d").length, route: e.querySelector(".ride").getAttribute("d").length})')}
        if 'Generated route' in status:
            with page.expect_download() as download:
                page.get_by_text('download GPX ↗', exact=True).click()
            path = output / f'text-{args.prompt}.gpx'
            download.value.save_as(str(path))
            points = gpxpy.parse(path.read_text()).tracks[0].segments[0].points
            evidence['gpx_points'] = len(points)
            evidence['bounds'] = [min(q.latitude for q in points), min(q.longitude for q in points),
                                  max(q.latitude for q in points), max(q.longitude for q in points)]
            assert len(points) > 2
            assert all(37.69 < q.latitude < 37.82 and -122.53 < q.longitude < -122.35 for q in points)
            assert evidence['map']['roads'] > 10000 and evidence['map']['route'] > 20
            assert evidence['map']['viewBox'] != '0 0 511 511'
            assert 'mi' in status and not errors
        (output / f'text-{args.prompt}.json').write_text(json.dumps(evidence, indent=2))
        browser.close()
        assert 'Generated route' in status, status


if __name__ == '__main__':
    main()
