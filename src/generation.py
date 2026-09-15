"""Description → vetted vector outline → bounded directed street search → GPX."""
import logging
import threading
import time
from dataclasses import asdict, dataclass, field, replace

import gpxpy.gpx
import numpy as np
from PIL import Image, ImageDraw
from src.geo import SF, TO_GEO
from src.graph import edge_coordinates
from src.street_search import StreetSearch

# The website is for rides people actually take. The historical 3–15 km acceptance band
# discarded most qualifying loops — measured on the catalogue, umbrella and heart found
# nothing at all while placements were rejected purely for falling outside it.
# Raising the ceiling alone changes nothing, because the ladder decides how long a loop
# can even be proposed: measured over seven catalogue shapes, 2–50 km with the old
# 3–14 km ladder still produced a 20.1 km mean, while adding the 17 and 20 km rungs
# lifted it to 27.8 km (max 29.7) with all seven still routing inside the 8 s budget.
# SF is only ~10 km across, so 50 km is headroom rather than a promise.
# Experiment runners keep the historical defaults.
WEBSITE_DISTANCE_RANGE = (2000, 50000)
WEBSITE_PERIMETERS = (3000, 5000, 8000, 11000, 14000, 17000, 20000)
# Riders asked for longer rides, so the ladder is walked from the top and the
# longest qualifying loop is offered first.
WEBSITE_PREFERENCE = 'longest'
from src.diffusion import DiffusionGenerator
from src.raster_outline import extract_outline

logger = logging.getLogger(__name__)


def candidate_record(candidate):
    """JSON-safe provenance; binary pixels live in the gallery/cache, not JSON."""
    from dataclasses import asdict
    row = asdict(candidate)
    row.pop('image', None)
    return row

@dataclass
class GenerationResult:
    image: Image.Image | None = None
    coordinates: list = field(default_factory=list)
    distance_m: float | None = None
    gpx: str | None = None
    status: str = ''
    timings: list = field(default_factory=list)
    debug_images: list = field(default_factory=list)
    outline: object = None
    route_xy: object = None
    total_seconds: float = 0
    interpretation: str = ''
    search_result: object = None
    alternative_index: int = 0
    prompt: str = ''
    diagnostics: dict = field(default_factory=dict)
    selected_candidate: int | None = None
    selected_outline: object = None


def render_outline(spec):
    image = Image.new('RGB',(512,512),'#fafafa')
    draw = ImageDraw.Draw(image)
    p = np.asarray(spec.points)*430+41
    draw.polygon([tuple(q) for q in p],fill='#637baa')
    return image


def render_streets(graph, frame=SF, route=None, conditioning=False):
    image = Image.new("RGB", (frame.size, frame.size), "black" if conditioning else "#f7f5ef")
    draw = ImageDraw.Draw(image)
    for u, v, data in graph.edges(data=True):
        xy = edge_coordinates(graph, u, v, data)
        pixels = [tuple(p) for p in frame.meters_to_pixels(xy)]
        draw.line(pixels, fill="white" if conditioning else "#c8ccc6", width=1)
    if route is not None:
        pixels = [tuple(p) for p in frame.meters_to_pixels(route)]
        draw.line(pixels, fill="#d75332", width=3)
        x, y = pixels[0]
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="#183e32", outline="white")
    return image


def to_gpx(coordinates, prompt):
    gpx = gpxpy.gpx.GPX()
    gpx.name = f"Route Sculptor: {prompt}"
    gpx.description = "SF bicycle street loop. © OpenStreetMap contributors. Check current conditions before riding."
    track = gpxpy.gpx.GPXTrack(name=gpx.name)
    segment = gpxpy.gpx.GPXTrackSegment()
    segment.points = [gpxpy.gpx.GPXTrackPoint(lat, lon) for lat, lon in coordinates]
    track.segments.append(segment)
    gpx.tracks.append(track)
    return gpx.to_xml()



class RouteService:
    def __init__(self, generator=None, search=None, frame=SF):
        # The public product intentionally starts from generated pixels.  Catalogue and
        # vector generators can still be injected by experimental callers.
        self.generator = generator if generator is not None else DiffusionGenerator()
        self.search = search
        self.frame = frame
        self.lock = threading.Lock()

    def stream(self, prompt):
        started = time.perf_counter()
        prompt = (prompt or '').strip()
        result = GenerationResult(prompt=prompt)
        if not prompt or len(prompt) > 200:
            result.status = 'Describe a shape in 1–200 characters.'
            yield result
            return
        with self.lock:
            stage = 'interpretation'
            if hasattr(self.generator, 'candidates'):
                yield from self.stream_candidates(prompt, started)
                return
            try:
                spec, timings = self.generator.generate(prompt)
                result.interpretation = spec.interpretation
                result.image = render_outline(spec)
                result.debug_images = [(result.image,'Interpreted silhouette')]
                result.diagnostics['interpretation'] = timings
                result.timings.append({'stage':'Interpret description','seconds':round(time.perf_counter()-started,3),'status':'done'})
                result.status = f'Interpreted as {spec.interpretation}. Searching SF streets…'
                # Yield before loading/searching streets so the preview is immediately visible.
                yield replace(result,total_seconds=round(time.perf_counter()-started,3))
                stage = 'street search'
                then = time.perf_counter()
                if self.search is None:
                    self.search = StreetSearch.load_prepared(frame=self.frame, perimeters=WEBSITE_PERIMETERS,
                                                   distance_range=WEBSITE_DISTANCE_RANGE,
                                                   prefer=WEBSITE_PREFERENCE)
                result.timings.append({'stage':'Load search index','seconds':round(time.perf_counter()-then,3),'status':'done'})
                found = self.search.search(spec)
                result.search_result = found
                result.diagnostics['search'] = found.diagnostics
                result.diagnostics['placements'] = found.placements
                result.diagnostics['search_timings'] = found.timings
                result.timings.append({'stage':'Search streets','seconds':round(found.timings.get('search_seconds',found.timings.get('cache_seconds',0)),3),'status':'done' if found.routes else 'failed'})
                if found.routes:
                    result = self.select_route(result,0)
                else:
                    result.status = f'Interpreted as {spec.interpretation}. {found.failure_reason} Your outline is kept above.'
            except ValueError as exc:
                result.status = str(exc)
            except Exception:
                logger.exception('Route generation failed')
                result.status = ('The shape interpreter is unavailable right now.' if stage == 'interpretation'
                                 else 'Street search is unavailable right now. Your outline is kept above.')
            result.total_seconds = round(time.perf_counter()-started,3)
            yield result

    def stream_candidates(self, prompt, started):
        from src.outlines import OutlineSpec
        records, options = [], []
        result = GenerationResult(prompt=prompt, interpretation=prompt)
        for candidate in self.generator.candidates(prompt):
            records.append(candidate)
            # Diffusion candidates are untrusted raster images.  Extract exactly one
            # validated outer silhouette before the historical route search sees it.
            if hasattr(candidate, 'image'):
                if candidate.image is None:
                    continue
                extracted = extract_outline(candidate.image, prompt)
                candidate.outline = (asdict(extracted.outline) if extracted.outline else None)
                candidate.extraction = extracted.diagnostics
                candidate.extraction_error = extracted.reason
                if extracted.outline is None:
                    result = replace(result, image=candidate.image,
                                     debug_images=[(candidate.image, 'Generated silhouette')],
                                     status='Generated image could not be converted to a routeable outline: ' + extracted.reason)
                    yield replace(result, diagnostics={'candidates':[candidate_record(c) for c in records]})
                    continue
                spec = extracted.outline
                # The extracted specification is in normalized drawing space.
                # Only select_route can supply a placed outline in map meters.
                result = replace(result, image=render_outline(spec), outline=None,
                    debug_images=[(candidate.image, 'Generated silhouette'), (extracted.mask, 'Extracted mask'),
                                  (extracted.preview, 'Validated outer contour')],
                    status='Generated silhouette. Searching SF streets…')
            else:
                if candidate.outline is None:
                    continue
                spec = OutlineSpec.parse(candidate.outline)
                result = replace(result, image=render_outline(spec), status='Generated silhouette. Searching streets…')
            yield replace(result, diagnostics={'candidates':[candidate_record(c) for c in records]})
            try:
                if self.search is None:
                    self.search = StreetSearch.load_prepared(frame=self.frame, perimeters=WEBSITE_PERIMETERS,
                                                   distance_range=WEBSITE_DISTANCE_RANGE,
                                                   prefer=WEBSITE_PREFERENCE)
                    from pathlib import Path
                    self.search.route_cache_dir = Path('route_cache/generative-v1')
                found = self.search.search(spec)
                candidate.timings.update(found.timings)
                if found.routes:
                    candidate.route = found.routes[0]
                    # Promotion requires a separately configured fidelity judge.
                    judge = getattr(self.generator, 'judge', None)
                    if judge is not None:
                        candidate.scores.update(judge(candidate))
                    if judge is not None and candidate.scores.get('excluded'):
                        candidate.infrastructure_error = candidate.scores.get('error', 'Judge unavailable')
                        continue
                    fidelity = candidate.scores.get('route_prompt_fidelity')
                    rank = (-(fidelity if fidelity is not None else 0), found.routes[0]['scores']['rank'])
                    options.append((rank, found, result.image, result.debug_images))
            except Exception as exc:
                candidate.infrastructure_error = str(exc)
        if options:
            _, found, preview, debug_images = min(options, key=lambda item:item[0])
            result = replace(result, search_result=found, image=preview, debug_images=debug_images)
            result = self.select_route(result, 0)
            result.status = 'Generated route · ' + result.status
        else:
            generation_errors = [getattr(c, 'error', None) for c in records if getattr(c, 'error', None)]
            if records and len(generation_errors) == len(records):
                # Keep the provider failure visible in the run details and status.  A
                # generic "no route" message makes an unavailable worker look like a
                # valid silhouette that simply did not fit the map.
                result.status = 'Text generation failed: ' + generation_errors[0]
            else:
                result.status = 'No qualifying generated street route. The generated image is retained; no catalogue replacement was used.'
        result.diagnostics['candidates'] = [candidate_record(c) for c in records]
        result.total_seconds = time.perf_counter()-started
        yield result

    def generate_outlines(self, prompt):
        """Generate and validate silhouettes without spending time on street search."""
        prompt = (prompt or '').strip()
        result = GenerationResult(prompt=prompt, interpretation=prompt)
        records, previews, debug = [], [], []
        if not prompt or len(prompt) > 200:
            result.status = 'Describe a shape in 1–200 characters.'
            return result
        for candidate in self.generator.candidates(prompt):
            candidate_index = len(records) + 1
            if candidate.image is not None:
                extracted = extract_outline(candidate.image, prompt)
                candidate.outline = asdict(extracted.outline) if extracted.outline else None
                candidate.extraction = extracted.diagnostics
                candidate.extraction_error = extracted.reason
                debug.append((candidate.image, f'Candidate {candidate_index} generated'))
                if extracted.outline is not None:
                    # Preview the validated contour, rather than asking the user to
                    # select from an untrusted raster that may contain scenery/text.
                    previews.append((extracted.preview, f'Candidate {candidate_index}'))
                    debug.extend([(extracted.mask, f'Candidate {candidate_index} mask'),
                                  (extracted.preview, f'Candidate {candidate_index} validated contour')])
            records.append(candidate)
        valid_records = [c for c in records if c.outline is not None]
        valid = list(range(len(valid_records)))
        result.image = previews[0][0] if previews else None
        result.debug_images = debug
        result.diagnostics = {'candidates': [candidate_record(c) for c in valid_records],
                              'all_candidates': [candidate_record(c) for c in records],
                              'valid_candidates': valid, 'outline_previews': previews}
        result.status = (f'{len(valid)} candidate' + ('s' if len(valid) != 1 else '') + '. Select one, then fit it to streets.'
                         if valid else 'No valid silhouette was generated. Try a more specific description; no catalogue substitute is available.')
        return result

    def fit_selected(self, result, selected_candidate):
        """Fit only the explicitly selected validated candidate to the street graph."""
        from src.outlines import OutlineSpec
        if result is None or not result.diagnostics:
            raise ValueError('Generate silhouettes before fitting a route.')
        try:
            index = int(selected_candidate)
            record = result.diagnostics['candidates'][index]
            outline = record.get('outline')
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise ValueError('Choose a valid silhouette before fitting a route.') from exc
        if outline is None:
            raise ValueError('That generated silhouette is invalid and cannot be routed.')
        spec = OutlineSpec.parse(outline)
        if self.search is None:
            self.search = StreetSearch.load_prepared(frame=self.frame, perimeters=WEBSITE_PERIMETERS,
                                                   distance_range=WEBSITE_DISTANCE_RANGE,
                                                   prefer=WEBSITE_PREFERENCE)
        found = self.search.search(spec)
        result.selected_candidate = index
        result.selected_outline = outline
        result.diagnostics = {**result.diagnostics, 'selected_candidate': index,
                              'selected_outline': outline, 'search': found.diagnostics,
                              'placements': found.placements, 'search_timings': found.timings}
        result.search_result = found
        if found.routes:
            fitted = self.select_route(result, 0)
            fitted.status = 'Generated route · ' + fitted.status
            return fitted
        result.status = f'No qualifying street route for selected silhouette. {found.failure_reason}'
        return result

    def generate(self,prompt):
        return list(self.stream(prompt))[-1]

    def select_route(self,result,index):
        if result.search_result is None or not result.search_result.routes:
            return result
        index %= len(result.search_result.routes)
        route = result.search_result.routes[index]
        xy = np.asarray(route['xy'])
        t = route['transform']
        outline = self.search.transform(result.search_result.outline,t['x'],t['y'],t['angle'],t['perimeter'])
        lon,lat = TO_GEO.transform(xy[:,0],xy[:,1])
        coordinates = list(zip(map(float,lat),map(float,lon)))
        status = f"{route['distance_m']/1609.344:.1f} mi loop · {result.interpretation}"
        debug_map = Image.new('RGB',(512,512),'#fafafa')
        draw = ImageDraw.Draw(debug_map)
        for placement in result.search_result.placements:
            x,y = self.frame.meters_to_pixels([[placement['x'],placement['y']]])[0]
            draw.ellipse((x-2,y-2,x+2,y+2),fill='#c6c9c3')
        draw.line([tuple(q) for q in self.frame.meters_to_pixels(outline)],fill='#cc5966',width=2)
        draw.line([tuple(q) for q in self.frame.meters_to_pixels(xy)],fill='#234dab',width=2)
        existing_debug = result.debug_images or [(result.image, 'Generated silhouette')]
        return replace(result,route_xy=xy,outline=outline,coordinates=coordinates,
                       gpx=to_gpx(coordinates,result.prompt),distance_m=route['distance_m'],status=status,
                       alternative_index=index,debug_images=[*existing_debug,(debug_map,'Candidate centers, placed outline and street loop')],diagnostics={**result.diagnostics,'chosen_transform':t,'scores':route['scores'],'directed_edges':route['edges']})
