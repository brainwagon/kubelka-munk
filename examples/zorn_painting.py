"""Repaint a photograph as an oil painting, in the Zorn palette.

Where ``zorn_filter.py`` replaces each pixel independently, this lays down brush strokes:
big ones first to block the picture in, then progressively smaller ones, and only where
the painting so far is still visibly wrong. It is a simplified version of Aaron
Hertzmann's "Painterly Rendering with Curved Brush Strokes of Multiple Sizes"
(SIGGRAPH 1998), whose central observation is that a painter does not work at one scale
uniformly -- the background gets three strokes and an eye gets thirty.

Each layer works like this:

  1. Blur the photograph by about the width of the brush. A big brush cannot see detail,
     so it should not be asked to reproduce any.
  2. Compare that against the painting so far, and divide the canvas into cells the size
     of the brush. Where a cell is close enough already, leave it alone -- this is what
     stops small brushes from stippling flat areas of sky.
  3. Everywhere else, start a stroke at the worst pixel in the cell.
  4. Grow the stroke along the direction the image is *not* changing, so strokes run
     along edges and around forms rather than across them. Stop when the stroke wanders
     somewhere it no longer belongs.

Three things separate this from the same algorithm drawing coloured worms.

The direction a stroke travels comes from a *structure tensor*, not from the raw gradient.
The gradient at a point is noisy, flips sign across a ridge, and in a flat region says
whatever the noise says; strokes built on it wiggle constantly. The structure tensor
averages the gradient's outer product over a neighbourhood, which is stable, and it
reports how much it should be believed, so flat passages can be given calm parallel
strokes instead of squiggles.

A mark is drawn as several tapered bristle tracks rather than one constant-width band
with a round cap, which is the shape a mouse makes.

And the paint is given thickness. Strokes accumulate into a height field which is lit
from the side at the end, so the ridge along each stroke catches the light. That is most
of what tells the eye it is looking at paint rather than at a picture of a colour.

Stroke colours are snapped to mixtures of the four Zorn paints, so the result is a
picture that could in principle be painted with four tubes. Pass ``--mixtures N`` to
restrict it further to the N mixtures that suit this picture best.

Run it with:  python examples/zorn_painting.py <image> [output] [-m N] [--seed N]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, sobel
from scipy.spatial import cKDTree

from kubelka_munk import srgb_to_xyz, xyz_to_lab
from zorn_filter import build_chit_colours, choose_best_mixtures
from zorn_wheel import build_palette

# Brush widths in pixels, coarsest first. Each is half the last, which is what makes the
# picture arrive as successive refinements rather than as one flat pass.
BRUSH_RADII = [32, 16, 8, 4]

# How blurred the reference is for a given brush, as a multiple of its radius.
BLUR_PER_RADIUS = 0.5

# A cell is repainted when its average colour difference exceeds this. Raising it leaves
# more of the coarse underpainting showing; lowering it drives the picture towards the
# photograph and away from looking painted.
REPAINT_THRESHOLD = 8.0

# Stroke lengths, in brush radii.
MINIMUM_STROKE = 4
MAXIMUM_STROKE = 12

# How readily a stroke changes direction: 1 follows the orientation field exactly, 0 never
# turns at all. This is Hertzmann's curvature filter, kept low because a stroke that can
# turn freely at every step wanders instead of travelling.
CURVATURE = 0.25

# Below this coherence -- how strongly oriented the picture is at a point, 0 for a flat
# region and 1 for a clean edge -- strokes stop following the image and fall back to a
# common direction. Flat passages are exactly where the orientation is meaningless noise,
# and where a painter would lay calm parallel strokes rather than squiggles.
COHERENCE_FLOOR = 0.15
COHERENCE_CEILING = 0.5
FALLBACK_ANGLE = 40.0

# Coherence is a ratio of eigenvalues, so it is blind to scale: the faintest imaginable
# texture, if consistently oriented, scores as highly as a clean edge. A blown-out sky is
# full of such texture, and following it produces swirls in what should be calm paint.
# So the field is trusted only where there is real contrast as well.
#
# That contrast has to be judged locally, not against the picture as a whole. A face is
# far lower in contrast than a row of buildings behind it, so any single global threshold
# either follows the buildings and hatches the face flat, or follows the face and turns
# the sky into swirls. Comparing each point against the average energy of its own
# neighbourhood separates them cleanly -- on this photograph, blown sky scores 0.00, a
# flat cheek 0.16, and eyes and brows 0.69, despite the last being a tenth the contrast
# of the buildings.
ENERGY_NEIGHBOURHOOD = 12.0
ENERGY_FLOOR = 0.5

# A mark is drawn as several parallel bristle tracks rather than one solid band, tapering
# towards the end as the brush lifts. This is most of what separates a brush from a marker.
BRISTLES = 3
BRISTLE_SPREAD = 0.8
TAPER_TO = 0.35
WIDTH_VARIATION = (0.75, 1.15)

# Paint stands off the canvas and its ridges catch the light. Strokes accumulate into a
# height field which is then lit from the upper left.
IMPASTO_STRENGTH = 1.6
IMPASTO_BLUR = 1.1
LIGHT_FROM = (-0.7071, -0.7071)

# Video of the painting being made.
VIDEO_SECONDS = 15.0
VIDEO_FPS = 30
VIDEO_HOLD_SECONDS = 1.5

# How the running time is shared between layers. Sharing it in proportion to the number of
# strokes would hand almost the whole film to the last layer, which has forty times as many
# strokes as the first and the least to show for them. A fractional power evens out how
# much the picture visibly changes per second.
FRAME_WEIGHT = 0.3

# How far a stroke's colour may wander from the palette mixture it started with before it
# is cut short, in units of colour difference.
STROKE_TOLERANCE = 25.0

# How much each stroke's colour is nudged before it is snapped to a mixture, in units of
# colour difference. A real brush is never loaded twice with quite the same colour, and a
# passage painted in one flat tint looks printed rather than painted. Setting this to
# zero turns the jitter off.
COLOUR_JITTER = 5.0


def paint(
    image: np.ndarray,
    palette_colours: np.ndarray,
    radii: list[int] | None = None,
    seed: int = 0,
    jitter: float = COLOUR_JITTER,
    recorder: "Recorder | None" = None,
) -> Image.Image:
    """Work from the coarsest brush to the finest, refining what the last one missed."""
    radii = BRUSH_RADII if radii is None else radii
    height, width, _ = image.shape
    generator = np.random.default_rng(seed)

    snap = _palette_snapper(palette_colours)

    # Start from the picture's average colour, so that anywhere the brushes never visit
    # still reads as part of the painting rather than as a hole.
    average = snap(xyz_to_lab(srgb_to_xyz(image.mean(axis=(0, 1))))[np.newaxis])[0]
    canvas = Image.new("RGB", (width, height), tuple(average.tolist()))

    # How thick the paint is, everywhere. Later strokes overwrite earlier ones, which is
    # what paint does.
    relief = Image.new("L", (width, height), 128)

    # The layers have to be planned before any are painted, so the film's running time
    # can be shared out between them.
    planned = []
    scratch = canvas.copy()
    for radius in radii:
        reference = gaussian_filter(
            image, sigma=(BLUR_PER_RADIUS * radius, BLUR_PER_RADIUS * radius, 0)
        )
        strokes = _plan_layer(np.asarray(scratch) / 255.0, reference, radius)
        generator.shuffle(strokes)
        planned.append((radius, reference, strokes))
        # A rough stand-in for what the layer will do, good enough to plan the next one.
        scratch = _preview_layer(scratch, reference, strokes, radius)

    budget = (
        _plan_frames(
            [len(strokes) for _, _, strokes in planned],
            round((VIDEO_SECONDS - VIDEO_HOLD_SECONDS) * VIDEO_FPS),
        )
        if recorder
        else [0] * len(planned)
    )

    if recorder:
        recorder.frame(canvas, relief)

    for (radius, reference, strokes), frames in zip(planned, budget):
        _draw_layer(
            canvas, relief, strokes, reference, radius, snap, generator, jitter,
            recorder, frames,
        )
        print(f"    brush {radius:3d}px  {len(strokes):6d} strokes")

    if recorder:
        recorder.frame(canvas, relief, times=round(VIDEO_HOLD_SECONDS * VIDEO_FPS))

    return _apply_impasto(canvas, relief)


class Recorder:
    """Pipes frames of the painting-in-progress straight into ffmpeg.

    Frames are handed over as raw pixels rather than encoded images, so nothing touches
    the disk until ffmpeg writes the finished file.
    """

    def __init__(self, path: Path, size: tuple[int, int], fps: int = VIDEO_FPS) -> None:
        width, height = size
        self.size = size
        # H.264 in the widely-playable pixel format needs even dimensions, and an odd
        # image would otherwise fail deep inside ffmpeg with nothing useful said about it.
        self.process = subprocess.Popen(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{width}x{height}", "-r", str(fps),
                "-i", "-",
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                "-pix_fmt", "yuv420p",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )
        self.count = 0

    def frame(self, painting: Image.Image, relief: Image.Image, times: int = 1) -> None:
        """Record the painting as it currently stands, lit as the finished one will be."""
        pixels = _apply_impasto(painting, relief).tobytes()
        for _ in range(times):
            self.process.stdin.write(pixels)
            self.count += 1

    def close(self) -> None:
        self.process.stdin.close()
        self.process.wait()


def _plan_frames(counts: list[int], total: int) -> list[int]:
    """Share the film's running time between the layers.

    Not in proportion to the number of strokes, which would give the whole film to the
    last layer -- there are forty times as many strokes in it as in the first, and the
    least to show for them, since by then the picture is only being refined. Weighting by
    a fractional power leaves the fine work dominant while keeping the blocking-in long
    enough to watch, which is the part worth seeing.
    """
    weights = [count ** FRAME_WEIGHT for count in counts]
    scale = total / max(sum(weights), 1e-9)
    return [max(1, round(weight * scale)) for weight in weights]


def _orientation_field(reference: np.ndarray, radius: int) -> np.ndarray:
    """Which way the strokes should run, everywhere, as a field of unit vectors.

    Taking the gradient at a point and turning ninety degrees gives a direction, but a
    noisy one: it flips sign across a ridge, and in a flat region it is whatever the
    noise happens to say. Strokes built from it wander.

    The structure tensor fixes this. It is the outer product of the gradient with itself,
    averaged over a neighbourhood -- and because it is the *tensor* that gets averaged
    rather than the angles, opposing gradients reinforce instead of cancelling. Its minor
    eigenvector is the direction along which the picture changes least, which is the
    direction a stroke should travel: along an edge rather than across it.

    It also yields a confidence for free. The gap between the two eigenvalues, relative to
    their sum, says how strongly oriented a neighbourhood really is. Where that is small
    the image has no opinion, and imposing one produces the scribble this replaces, so
    those regions are handed a single common direction instead.

    That ratio is scale-blind, though, and will happily report a confident direction for
    texture far too faint to see. So it is multiplied by the gradient energy, judged
    against the energy of the surrounding neighbourhood: a region must be both
    consistently oriented *and* have more going on than its surroundings before the
    strokes will follow it.
    """
    luminance = reference @ np.array([0.2126, 0.7152, 0.0722])
    gradient_x = sobel(luminance, axis=1)
    gradient_y = sobel(luminance, axis=0)

    spread = max(1.5, float(radius))
    xx = gaussian_filter(gradient_x * gradient_x, spread)
    xy = gaussian_filter(gradient_x * gradient_y, spread)
    yy = gaussian_filter(gradient_y * gradient_y, spread)

    difference = xx - yy
    root = np.sqrt(difference * difference + 4.0 * xy * xy)
    total = xx + yy
    coherence = np.divide(root, total, out=np.zeros_like(root), where=total > 1e-12)

    # ...and how much contrast there is to be coherent about.
    neighbourhood = gaussian_filter(
        total, min(ENERGY_NEIGHBOURHOOD * radius, 200.0)
    )
    energy = np.clip(
        total / np.maximum(neighbourhood * ENERGY_FLOOR, 1e-12), 0.0, 1.0
    )

    # The major eigenvector points along the gradient; a quarter turn gives the tangent.
    angle = 0.5 * np.arctan2(2.0 * xy, difference)
    tangent = np.stack([-np.sin(angle), np.cos(angle)], axis=-1)

    fallback = np.array(
        [np.cos(np.radians(FALLBACK_ANGLE)), np.sin(np.radians(FALLBACK_ANGLE))]
    )

    # A tangent has no inherent sign, so point them all the same way. This alone stops
    # neighbouring strokes running head-on into each other.
    facing = np.sign(tangent @ fallback)
    tangent *= np.where(facing == 0, 1.0, facing)[..., np.newaxis]

    trust = np.clip(
        (coherence - COHERENCE_FLOOR) / (COHERENCE_CEILING - COHERENCE_FLOOR), 0.0, 1.0
    )
    trust = (trust * energy)[..., np.newaxis]
    field = trust * tangent + (1.0 - trust) * fallback

    length = np.linalg.norm(field, axis=-1, keepdims=True)
    return field / np.maximum(length, 1e-9)


def _plan_layer(
    canvas: np.ndarray, reference: np.ndarray, radius: int
) -> list[tuple[int, int]]:
    """Where this brush is needed: the worst pixel of every cell that is not good enough.

    One stroke per cell, sized to the brush, is what keeps the strokes roughly evenly
    spaced without any explicit spacing rule.
    """
    difference = np.linalg.norm(
        xyz_to_lab(srgb_to_xyz(canvas)) - xyz_to_lab(srgb_to_xyz(reference)), axis=-1
    )

    height, width = difference.shape
    cells_down = -(-height // radius)
    cells_across = -(-width // radius)

    # Pad out to whole cells so the grid can be reshaped rather than looped over. The
    # padding is worthless data, so it is made unattractive to paint.
    padded = np.full((cells_down * radius, cells_across * radius), -1.0)
    padded[:height, :width] = difference
    blocks = padded.reshape(cells_down, radius, cells_across, radius).swapaxes(1, 2)

    flattened = blocks.reshape(cells_down, cells_across, radius * radius)
    needs_paint = flattened.mean(axis=2) > REPAINT_THRESHOLD

    worst = flattened.argmax(axis=2)
    cell_rows, cell_columns = np.nonzero(needs_paint)
    offsets = worst[cell_rows, cell_columns]

    rows = cell_rows * radius + offsets // radius
    columns = cell_columns * radius + offsets % radius

    inside = (rows < height) & (columns < width)
    return list(zip(columns[inside].tolist(), rows[inside].tolist()))


def _preview_layer(canvas, reference, strokes, radius):
    """Roughly what a layer will leave behind, for planning the next one.

    Painting a layer twice would double the running time, so this stands in: the strokes
    are known, and a blurred reference painted through discs of the right size is close
    enough to decide where the *next* brush will be needed.
    """
    preview = canvas.copy()
    stamp = ImageDraw.Draw(preview)
    colours = np.clip(np.round(reference * 255), 0, 255).astype(np.uint8)
    for x, y in strokes:
        stamp.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=tuple(colours[y, x].tolist()),
        )
    return preview


def _draw_layer(
    canvas, height, strokes, reference, radius, snap, generator, jitter,
    recorder=None, frames=0,
) -> None:
    """Trace and paint every stroke of one layer, and record how thick the paint got."""
    brush = ImageDraw.Draw(canvas)
    relief = ImageDraw.Draw(height)

    field = _orientation_field(reference, radius)
    reference_lab = xyz_to_lab(srgb_to_xyz(reference))

    columns = np.array([x for x, _ in strokes])
    rows = np.array([y for _, y in strokes])

    # Every colour for the whole layer in one lookup rather than one per bristle. Each
    # bristle is nudged separately, because a loaded brush does not carry one flat colour
    # across its width either.
    loaded = reference_lab[rows, columns]
    nudged = np.repeat(loaded[:, np.newaxis, :], BRISTLES, axis=1)
    if jitter:
        nudged = nudged + generator.normal(0.0, jitter, size=nudged.shape)
    colours = snap(nudged.reshape(-1, 3)).reshape(len(strokes), BRISTLES, 3)

    # How thickly each stroke is loaded. Varying it is what gives the lit height field
    # ridges between one stroke and the next.
    thickness = generator.integers(90, 210, size=len(strokes))
    widths = generator.uniform(*WIDTH_VARIATION, size=len(strokes))

    capture_every = max(1, len(strokes) // frames) if recorder and frames else 0

    for index, (x, y) in enumerate(strokes):
        points = _trace_stroke(
            x, y, radius, field, reference_lab, loaded[index]
        )
        width = radius * widths[index]
        tracks = _bristle_tracks(points, width)
        paint_height = int(thickness[index])

        if not tracks:
            # A stroke with nowhere to go is a dab, which is a legitimate mark.
            box = [x - width, y - width, x + width, y + width]
            brush.ellipse(box, fill=tuple(colours[index, 0].tolist()))
            relief.ellipse(box, fill=paint_height)
            if capture_every and index % capture_every == 0:
                recorder.frame(canvas, height)
            continue

        for bristle, outline in enumerate(tracks):
            brush.polygon(outline, fill=tuple(colours[index, bristle].tolist()))
            relief.polygon(outline, fill=paint_height)

        if capture_every and index % capture_every == 0:
            recorder.frame(canvas, height)


def _apply_impasto(painting: Image.Image, height: Image.Image) -> Image.Image:
    """Light the paint from the side, so its ridges show.

    Oil paint is not a flat film -- it stands off the canvas, and every stroke has an
    edge where it steps up from what is underneath. Those edges catch the light, and
    seeing them is most of what tells the eye it is looking at paint rather than at a
    picture of a colour.

    The height field is the record of how thickly each stroke was laid. Shading it by its
    own slope against a fixed light gives the highlight along one side of each ridge and
    the shadow along the other, with no need to model anything three-dimensional.
    """
    surface = gaussian_filter(np.asarray(height, dtype=float) / 255.0, IMPASTO_BLUR)
    slope_y, slope_x = np.gradient(surface)

    light_x, light_y = LIGHT_FROM
    shading = 1.0 + IMPASTO_STRENGTH * (slope_x * light_x + slope_y * light_y)
    shading = np.clip(shading, 0.55, 1.5)[..., np.newaxis]

    lit = np.asarray(painting, dtype=float) * shading
    return Image.fromarray(np.clip(lit, 0, 255).astype(np.uint8))


def _trace_stroke(
    x: int,
    y: int,
    radius: int,
    field: np.ndarray,
    reference_lab: np.ndarray,
    started_from: np.ndarray,
) -> list[tuple[float, float]]:
    """Follow the orientation field from a starting point.

    The stroke stops when it leaves the picture, or when it reaches somewhere whose
    colour no longer matches the colour the brush is loaded with -- a stroke should stay
    inside the form it began in.
    """
    height, width, _ = field.shape
    points = [(float(x), float(y))]
    position = np.array([float(x), float(y)])
    last_direction = np.zeros(2)

    for step in range(MAXIMUM_STROKE):
        column, row = int(round(position[0])), int(round(position[1]))
        if not (0 <= column < width and 0 <= row < height):
            break

        if step >= MINIMUM_STROKE:
            drifted = np.linalg.norm(reference_lab[row, column] - started_from)
            if drifted > STROKE_TOLERANCE:
                break

        direction = field[row, column]
        if direction @ last_direction < 0:
            direction = -direction

        if step > 0:
            direction = CURVATURE * direction + (1 - CURVATURE) * last_direction
            length = np.linalg.norm(direction)
            if length < 1e-9:
                break
            direction = direction / length

        position = position + radius * direction
        last_direction = direction
        points.append((float(position[0]), float(position[1])))

    return points


def _bristle_tracks(
    points: list[tuple[float, float]], radius: float
) -> list[list[tuple[float, float]]]:
    """Turn a path into a few tapered parallel ribbons, one per bristle.

    A constant-width band with a round cap is the shape a mouse makes, not a brush. Real
    bristles leave separate tracks with gaps between them, and the mark narrows as the
    brush lifts, so each track is built as a polygon whose width falls off along its
    length.
    """
    path = np.array(points, dtype=float)
    if len(path) < 2:
        return []

    # A direction at each point, averaged from the segments either side of it.
    segments = np.diff(path, axis=0)
    directions = np.zeros_like(path)
    directions[:-1] += segments
    directions[1:] += segments
    lengths = np.linalg.norm(directions, axis=1, keepdims=True)
    directions /= np.maximum(lengths, 1e-9)
    normals = np.stack([-directions[:, 1], directions[:, 0]], axis=1)

    along = np.linspace(0.0, 1.0, len(path))
    taper = 1.0 - (1.0 - TAPER_TO) * along

    offsets = (
        np.linspace(-1.0, 1.0, BRISTLES) if BRISTLES > 1 else np.zeros(1)
    ) * BRISTLE_SPREAD
    half_width = (radius / BRISTLES) * 1.15 * taper

    tracks = []
    for offset in offsets:
        centre = path + normals * (offset * radius)
        left = centre + normals * half_width[:, np.newaxis]
        right = centre - normals * half_width[:, np.newaxis]
        outline = np.vstack([left, right[::-1]])
        tracks.append([(float(px), float(py)) for px, py in outline])
    return tracks


def _palette_snapper(palette_colours: np.ndarray):
    """Maps CIELAB colours to the nearest mixtures, as RGB bytes ready for drawing.

    Works in CIELAB throughout, because that is the space the strokes are chosen and
    jittered in, and converting back and forth per stroke would be both slower and
    slightly lossy.
    """
    tree = cKDTree(xyz_to_lab(srgb_to_xyz(palette_colours)))
    as_bytes = np.clip(np.round(palette_colours * 255), 0, 255).astype(np.uint8)

    def snap(lab: np.ndarray) -> np.ndarray:
        """Takes an (n, 3) array of CIELAB colours, returns (n, 3) of RGB bytes."""
        _, indices = tree.query(np.atleast_2d(lab))
        return as_bytes[indices]

    return snap


def main(
    source: str,
    destination: str | None = None,
    mixtures: int | None = None,
    seed: int = 0,
    longest_side: int = 1400,
    radii: list[int] | None = None,
    jitter: float = COLOUR_JITTER,
    video: str | None = None,
) -> None:
    source_path = Path(source)
    destination_path = (
        Path(destination)
        if destination
        else source_path.with_name(source_path.stem + "_painting.png")
    )

    original = Image.open(source_path).convert("RGB")
    if max(original.size) > longest_side:
        scale = longest_side / max(original.size)
        original = original.resize(
            (round(original.width * scale), round(original.height * scale)),
            Image.LANCZOS,
        )

    image = np.asarray(original, dtype=float) / 255.0
    print(f"\n  {source_path.name}: painting at {image.shape[1]} x {image.shape[0]}")

    palette = build_palette()
    palette_colours, weights = build_chit_colours(palette)
    if mixtures is not None:
        keep = choose_best_mixtures(image, palette_colours, mixtures)
        palette_colours = palette_colours[keep]
        print(f"  restricted to {len(keep)} mixtures")

    recorder = None
    if video is not None:
        if shutil.which("ffmpeg") is None:
            raise SystemExit("ffmpeg is needed to record the painting, and is not on PATH")
        video_path = Path(video)
        recorder = Recorder(video_path, (image.shape[1], image.shape[0]))
        print(f"  recording {VIDEO_SECONDS:.0f}s at {VIDEO_FPS}fps to {video_path}")

    painting = paint(
        image, palette_colours, radii=radii, seed=seed, jitter=jitter, recorder=recorder
    )
    painting.save(destination_path)

    if recorder is not None:
        recorder.close()
        print(f"  wrote {video_path} ({recorder.count / VIDEO_FPS:.1f}s, "
              f"{recorder.count} frames)")

    difference = np.linalg.norm(
        xyz_to_lab(srgb_to_xyz(image))
        - xyz_to_lab(srgb_to_xyz(np.asarray(painting, dtype=float) / 255.0)),
        axis=-1,
    )
    print(
        f"\n  difference from the photograph: median {np.median(difference):5.2f}"
        f"   mean {difference.mean():5.2f}"
    )
    print("  (a painting is meant to differ from its subject -- this is a measure of")
    print("   how loose the result is, not of how wrong it is)")
    print(f"\n  wrote {destination_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", help="the photograph to paint")
    parser.add_argument("destination", nargs="?", help="where to write the painting")
    parser.add_argument(
        "-m", "--mixtures", type=int, help="restrict the palette to this many mixtures"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="stroke order, for a different painting"
    )
    parser.add_argument(
        "--brushes",
        default=",".join(str(radius) for radius in BRUSH_RADII),
        help="brush radii in pixels, coarsest first (default %(default)s). Dropping the "
        "smallest gives a looser painting; adding one drives it towards the photograph",
    )
    parser.add_argument(
        "--jitter",
        type=float,
        default=COLOUR_JITTER,
        help="vary each stroke's colour by about this much before snapping it to a "
        "mixture (default %(default)s, 0 to disable)",
    )
    parser.add_argument(
        "--video",
        metavar="FILE.mp4",
        help=f"also record the painting being made, about {VIDEO_SECONDS:.0f} seconds long "
        "(needs ffmpeg)",
    )
    parser.add_argument(
        "--longest-side",
        type=int,
        default=1400,
        help="paint at this size (default 1400; brush sizes are in pixels)",
    )
    arguments = parser.parse_args()
    main(
        arguments.source,
        arguments.destination,
        arguments.mixtures,
        arguments.seed,
        arguments.longest_side,
        [int(radius) for radius in arguments.brushes.split(",")],
        arguments.jitter,
        arguments.video,
    )
