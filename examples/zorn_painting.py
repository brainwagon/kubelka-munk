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
from dataclasses import dataclass
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, gaussian_filter, sobel, zoom
from scipy.spatial import cKDTree

from kubelka_munk import srgb_to_xyz, xyz_to_lab
from zorn_filter import build_chit_colours, choose_best_mixtures
from zorn_wheel import build_palette

# Brush widths in pixels, coarsest first. Each is half the last, which is what makes the
# picture arrive as successive refinements rather than as one flat pass.
BRUSH_RADII = [32, 16, 8, 4]

# How blurred the reference is for a given brush, as a multiple of its radius.
BLUR_PER_RADIUS = 0.5

# Stroke lengths, in brush radii. The maximum is set by the style, below.
MINIMUM_STROKE = 4

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

# Where the picture has no opinion about direction, the strokes hatch at this angle --
# but not all at exactly this angle. One fixed direction across a whole canvas is a
# rendering convention showing through; a painter's hatching wanders with the form and
# with the reach of their arm. So the angle drifts by up to this much over the picture,
# smoothly, on a scale of a few hundred pixels.
FALLBACK_ANGLE = 40.0
FALLBACK_DRIFT = 25.0
FALLBACK_SCALE = 220.0

# How confident the orientation has to be before a stroke follows it rather than hatching.
# This is a threshold rather than a blend on purpose: blending a measured direction
# towards the hatching angle produces directions that are neither, and biases the whole
# picture towards the diagonal.
TRUST_TO_FOLLOW = 0.5

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
WIDTH_VARIATION = (0.75, 1.15)

# The mark narrows to nearly nothing as the brush lifts, rather than stopping at a third
# of its width and being cut off square.
TAPER_TO = 0.12

# Bristles do not all land and lift together, so each track starts and finishes somewhere
# within this fraction of either end of the path. Without it every bristle stops on the
# same line and the stroke ends look guillotined however they are shaped.
BRISTLE_STAGGER = 0.25

# Points around the rounded end where the brush first touches down.
CAP_POINTS = 6

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

# How much looser the background is painted than the subject, when the two are separated.
# Keeping the background broader is how a painter directs attention: detail is expensive,
# and spending it evenly over a picture is part of what makes one read as a photograph.
BACKGROUND_LOOSER_BY = 0.25

# Filling the subject's area in so there is a background to paint behind him: how hard,
# and how many times, the nearest-neighbour fill is smoothed to relax its streaks.
INPAINT_SMOOTHING = 9.0
INPAINT_RELAXATIONS = 12

# The one dial for how tightly the painting follows the photograph. Half way reproduces
# the settings these were tuned to by hand; see Style below for what it moves.
DEFAULT_TIGHTNESS = 0.5


def paint(
    image: np.ndarray,
    palette_colours: np.ndarray,
    radii: list[int] | None = None,
    seed: int = 0,
    jitter: float | None = None,
    recorder: "Recorder | None" = None,
    fallback_angle: float = FALLBACK_ANGLE,
    style: "Style | None" = None,
    foreground: np.ndarray | None = None,
    background_style: "Style | None" = None,
) -> Image.Image:
    """Work from the coarsest brush to the finest, refining what the last one missed.

    Given a ``foreground`` mask the picture is painted in two passes rather than one: the
    whole background first, then the subject over the top. That is the order a painter
    works in, and it buys two things a single pass cannot. The background can be painted
    more broadly than the subject, which is how attention gets directed. And the
    subject's strokes are confined to its own silhouette, so the edge stays where it
    belongs instead of being chewed by strokes that began inside the face and ran out
    into the sky.

    The background's strokes are deliberately not confined. It is painted across the
    whole canvas, the subject's area included -- his shape is filled in beforehand with a
    plausible continuation of what surrounds it -- so the ground is complete before he is
    put in front of it. That is how the thing is actually done, and it matters here
    because the cut-out's edge is soft: paint the background only up to the silhouette
    and the bare canvas shows through along it as a halo.
    """
    style = Style.from_tightness(DEFAULT_TIGHTNESS) if style is None else style
    background_style = style if background_style is None else background_style
    height, width, _ = image.shape
    generator = np.random.default_rng(seed)

    snap = _palette_snapper(palette_colours)

    # One hatching field for the whole painting, so every layer agrees about which way to
    # lay paint where the picture does not say.
    hatching = _hatching_field((height, width), fallback_angle, generator)

    # Each pass names the region it takes its colours from, and the region it is allowed
    # to lay paint in. For the background those differ: it draws on the background only,
    # but paints the entire canvas, the subject's area included, so the ground is
    # complete before he is put in front of it.
    if foreground is None:
        passes = [("picture", None, None, style)]
    else:
        passes = [
            ("background", 1.0 - foreground, None, background_style),
            ("subject", foreground, foreground, style),
        ]

    # Start from the picture's average colour, so that anywhere the brushes never visit
    # still reads as part of the painting rather than as a hole.
    average = snap(xyz_to_lab(srgb_to_xyz(image.mean(axis=(0, 1))))[np.newaxis])[0]
    canvas = Image.new("RGB", (width, height), tuple(average.tolist()))

    # How thick the paint is, everywhere. Later strokes overwrite earlier ones, which is
    # what paint does.
    relief = Image.new("L", (width, height), 128)

    # Every layer of every pass has to be planned before any of it is painted, so that
    # the film's running time can be shared out between them.
    plans = []
    scratch = canvas.copy()
    for label, seen, paintable, pass_style in passes:
        subject = image if seen is None else _inpaint(image, seen)
        stencil = (
            None
            if paintable is None
            else Image.fromarray((paintable * 255).astype(np.uint8), "L")
        )
        for radius in pass_style.brushes(radii or BRUSH_RADII):
            reference = gaussian_filter(
                subject, sigma=(BLUR_PER_RADIUS * radius, BLUR_PER_RADIUS * radius, 0)
            )
            strokes = _plan_layer(
                np.asarray(scratch) / 255.0, reference, radius, pass_style, paintable
            )
            generator.shuffle(strokes)
            plans.append((label, stencil, pass_style, radius, reference, strokes))
            # A rough stand-in for what the layer will do, good enough to plan the next.
            scratch = _preview_layer(scratch, reference, strokes, radius, stencil)

    budget = (
        _plan_frames(
            [len(strokes) for *_, strokes in plans],
            round((VIDEO_SECONDS - VIDEO_HOLD_SECONDS) * VIDEO_FPS),
        )
        if recorder
        else [0] * len(plans)
    )

    if recorder:
        recorder.frame(canvas, relief)

    painted, painted_relief, mask, current = canvas, relief, None, None
    for (label, region_mask, pass_style, radius, reference, strokes), frames in zip(
        plans, budget
    ):
        if label != current:
            _stencil(canvas, relief, painted, painted_relief, mask)
            mask = region_mask
            # A masked pass is painted on its own copy and stencilled on afterwards, so
            # that its strokes cannot spill out past the silhouette.
            painted = canvas.copy() if mask is not None else canvas
            painted_relief = relief.copy() if mask is not None else relief
            current = label

        def capture(surface=painted, surface_relief=painted_relief, stencil=mask):
            """The painting as it would look if the pass in progress stopped here."""
            if stencil is None:
                recorder.frame(surface, surface_relief)
            else:
                preview, preview_relief = canvas.copy(), relief.copy()
                preview.paste(surface, (0, 0), stencil)
                preview_relief.paste(surface_relief, (0, 0), stencil)
                recorder.frame(preview, preview_relief)

        _draw_layer(
            painted,
            painted_relief,
            strokes,
            reference,
            radius,
            snap,
            generator,
            pass_style.jitter if jitter is None else jitter,
            hatching,
            pass_style,
            capture if recorder else None,
            frames,
        )
        print(f"    {label:10s} brush {radius:3d}px  {len(strokes):6d} strokes")

    _stencil(canvas, relief, painted, painted_relief, mask)

    if recorder:
        recorder.frame(canvas, relief, times=round(VIDEO_HOLD_SECONDS * VIDEO_FPS))

    return _apply_impasto(canvas, relief)


def _stencil(canvas, relief, painted, painted_relief, mask) -> None:
    """Lay a finished pass onto the painting, if it was painted on a copy of its own."""
    if mask is not None:
        canvas.paste(painted, (0, 0), mask)
        relief.paste(painted_relief, (0, 0), mask)


@dataclass(frozen=True)
class Style:
    """How tightly the painting follows the photograph.

    Looseness is not one setting but four moving together, which is why it is worth a
    dial of its own. A loose painting uses a big brush, lets a stroke run a long way
    before it stops, tolerates a patch being some way off before going back to it, and
    varies its colour freely. A tight one does the opposite on all four counts, and
    changing any one of them alone mostly just makes the picture worse.

    ``tightness`` runs from 0 to 1 and interpolates geometrically between the two, since
    every one of these is a scale rather than a position -- halfway between a threshold
    of 16 and one of 3.5 is 7.5, not 9.75. Half way reproduces the values these settings
    were hand-tuned to.
    """

    repaint_threshold: float
    maximum_stroke: int
    stroke_tolerance: float
    jitter: float
    brush_scale: float

    @classmethod
    def from_tightness(cls, tightness: float) -> "Style":
        tightness = float(np.clip(tightness, 0.0, 1.0))

        def between(loose: float, tight: float) -> float:
            return loose * (tight / loose) ** tightness

        return cls(
            # How wrong a patch has to be before the next brush goes back to it.
            repaint_threshold=between(16.0, 3.5),
            # How far a stroke may run, in brush radii.
            maximum_stroke=max(MINIMUM_STROKE + 1, round(between(18.0, 6.0))),
            # How far its colour may drift before it is cut short.
            stroke_tolerance=between(40.0, 12.0),
            # How much each stroke's colour is nudged before being snapped to a mixture.
            jitter=between(8.0, 2.5),
            # And how big the brushes are, against the sizes named in BRUSH_RADII.
            brush_scale=between(1.5, 0.65),
        )

    def brushes(self, radii: list[int]) -> list[int]:
        """The brush sizes this style actually paints with."""
        return [max(2, round(radius * self.brush_scale)) for radius in radii]

    def describe(self) -> str:
        return (
            f"brushes {self.brushes(BRUSH_RADII)}, repaint over "
            f"{self.repaint_threshold:.1f}, stroke <= {self.maximum_stroke}, "
            f"tolerance {self.stroke_tolerance:.0f}, jitter {self.jitter:.1f}"
        )


def foreground_mask(image: Image.Image, model: str = "u2net") -> np.ndarray:
    """Separate the subject from its background, as a soft mask in [0, 1].

    rembg is imported here rather than at the top of the file because it pulls in a
    neural network runtime and takes a noticeable moment to load, which nobody painting
    a picture in a single pass should have to wait for.
    """
    from rembg import new_session, remove

    cut_out = remove(image, session=new_session(model))
    return np.asarray(cut_out)[..., 3].astype(float) / 255.0


def _inpaint(image: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Fill everything outside the region with a plausible continuation of what is inside.

    Two reasons the background pass needs this. A blur does not respect a silhouette, so
    without it the subject's face would bleed outward and the background be painted in
    skin tones for a brush-width all around him. And the background is painted across the
    whole canvas, the subject included, so there has to be something behind him to paint.

    The method is about as simple as inpainting gets: fill each unknown pixel with its
    nearest known one, then repeatedly blur while holding the known pixels fixed. The
    first step fills everything immediately, and the second lets the fill relax from the
    hard radial streaks the nearest-neighbour pass leaves into something smooth. It
    invents nothing and knows nothing about structure, which is the right amount of
    ambition for paint that ends up underneath a portrait.
    """
    unknown = region < 0.5
    if not unknown.any():
        return image

    _, nearest = distance_transform_edt(unknown, return_indices=True)
    filled = image[nearest[0], nearest[1]]

    known = ~unknown
    for _ in range(INPAINT_RELAXATIONS):
        filled = gaussian_filter(filled, sigma=(INPAINT_SMOOTHING, INPAINT_SMOOTHING, 0))
        filled[known] = image[known]
    return filled


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


def _hatching_field(
    shape: tuple[int, int], angle: float, generator
) -> np.ndarray:
    """The direction to hatch in where the picture has nothing to say, as a field.

    A single angle everywhere reads as machinery. This wanders slowly across the canvas
    instead, by smoothing noise down to a very low frequency so that neighbouring strokes
    still agree while opposite corners of the picture need not.
    """
    height, width = shape
    coarse = generator.normal(size=(max(2, height // 64), max(2, width // 64)))
    spread = zoom(coarse, (height / coarse.shape[0], width / coarse.shape[1]), order=1)
    spread = gaussian_filter(spread[:height, :width], FALLBACK_SCALE)
    spread = spread / max(np.abs(spread).max(), 1e-9)

    angles = np.radians(angle + FALLBACK_DRIFT * spread)
    return np.stack([np.cos(angles), np.sin(angles)], axis=-1)


def _orientation_field(
    reference: np.ndarray, radius: int, hatching: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
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
    those regions are handed the hatching direction instead.

    That ratio is scale-blind, though, and will happily report a confident direction for
    texture far too faint to see. So it is multiplied by the gradient energy, judged
    against the energy of the surrounding neighbourhood: a region must be both
    consistently oriented *and* have more going on than its surroundings before the
    strokes will follow it.

    The tangents and the confidence are returned separately rather than blended together.
    Blending them rotates every middling-confidence direction towards the hatching angle,
    which quietly tilts the whole picture; the caller uses the confidence to *choose*
    instead.
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

    # A tangent has no inherent sign, so point them all the same way as the local
    # hatching. This alone stops neighbouring strokes running head-on into each other.
    facing = np.sign(np.sum(tangent * hatching, axis=-1))
    tangent = tangent * np.where(facing == 0, 1.0, facing)[..., np.newaxis]

    trust = (
        np.clip(
            (coherence - COHERENCE_FLOOR) / (COHERENCE_CEILING - COHERENCE_FLOOR),
            0.0,
            1.0,
        )
        * energy
    )
    return tangent, trust


def _plan_layer(
    canvas: np.ndarray,
    reference: np.ndarray,
    radius: int,
    style: Style,
    region: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """Where this brush is needed: the worst pixel of every cell that is not good enough.

    One stroke per cell, sized to the brush, is what keeps the strokes roughly evenly
    spaced without any explicit spacing rule.
    """
    difference = np.linalg.norm(
        xyz_to_lab(srgb_to_xyz(canvas)) - xyz_to_lab(srgb_to_xyz(reference)), axis=-1
    )

    if region is not None:
        # Outside this pass's region there is nothing to answer for, so a cell straddling
        # the boundary is judged only on the part that belongs to it, and the worst pixel
        # found within a cell is necessarily one of its own.
        difference = difference * region

    height, width = difference.shape
    cells_down = -(-height // radius)
    cells_across = -(-width // radius)

    # Pad out to whole cells so the grid can be reshaped rather than looped over. The
    # padding is worthless data, so it is made unattractive to paint.
    padded = np.full((cells_down * radius, cells_across * radius), -1.0)
    padded[:height, :width] = difference
    blocks = padded.reshape(cells_down, radius, cells_across, radius).swapaxes(1, 2)

    flattened = blocks.reshape(cells_down, cells_across, radius * radius)
    needs_paint = flattened.mean(axis=2) > style.repaint_threshold

    worst = flattened.argmax(axis=2)
    cell_rows, cell_columns = np.nonzero(needs_paint)
    offsets = worst[cell_rows, cell_columns]

    rows = cell_rows * radius + offsets // radius
    columns = cell_columns * radius + offsets % radius

    inside = (rows < height) & (columns < width)
    return list(zip(columns[inside].tolist(), rows[inside].tolist()))


def _preview_layer(canvas, reference, strokes, radius, region_mask=None):
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
    if region_mask is not None:
        stencilled = canvas.copy()
        stencilled.paste(preview, (0, 0), region_mask)
        return stencilled
    return preview


def _draw_layer(
    canvas, height, strokes, reference, radius, snap, generator, jitter, hatching,
    style, capture=None, frames=0,
) -> None:
    """Trace and paint every stroke of one layer, and record how thick the paint got."""
    brush = ImageDraw.Draw(canvas)
    relief = ImageDraw.Draw(height)

    tangent, trust = _orientation_field(reference, radius, hatching)
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

    capture_every = max(1, len(strokes) // frames) if capture and frames else 0

    for index, (x, y) in enumerate(strokes):
        points = _trace_stroke(
            x, y, radius, tangent, hatching, trust, reference_lab, loaded[index],
            style,
        )
        width = radius * widths[index]
        tracks = _bristle_tracks(points, width, generator)
        paint_height = int(thickness[index])

        if not tracks:
            # A stroke with nowhere to go is a dab, which is a legitimate mark.
            box = [x - width, y - width, x + width, y + width]
            brush.ellipse(box, fill=tuple(colours[index, 0].tolist()))
            relief.ellipse(box, fill=paint_height)
            if capture_every and index % capture_every == 0:
                capture()
            continue

        for bristle, outline in enumerate(tracks):
            brush.polygon(outline, fill=tuple(colours[index, bristle].tolist()))
            relief.polygon(outline, fill=paint_height)

        if capture_every and index % capture_every == 0:
            capture()


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
    tangent: np.ndarray,
    hatching: np.ndarray,
    trust: np.ndarray,
    reference_lab: np.ndarray,
    started_from: np.ndarray,
    style: Style,
) -> list[tuple[float, float]]:
    """Follow the picture from a starting point, as far as the picture is worth following.

    Confidence decides how readily the stroke *turns*, rather than what direction it
    points in. Where the orientation is trustworthy the stroke steers by it; where it is
    not, the stroke simply carries on the way it was already going. A brush behaves like
    this -- it travels, and the picture steers it more or less firmly -- and it means a
    passage with no direction in it gets straight strokes rather than strokes bent
    towards some house angle.

    The starting direction is chosen, not blended, for the same reason.
    """
    height, width, _ = tangent.shape
    points = [(float(x), float(y))]
    position = np.array([float(x), float(y)])
    last_direction = np.zeros(2)

    for step in range(style.maximum_stroke):
        column, row = int(round(position[0])), int(round(position[1]))
        if not (0 <= column < width and 0 <= row < height):
            break

        if step >= MINIMUM_STROKE:
            drifted = np.linalg.norm(reference_lab[row, column] - started_from)
            if drifted > style.stroke_tolerance:
                break

        confidence = trust[row, column]
        if step == 0:
            direction = (
                tangent[row, column]
                if confidence >= TRUST_TO_FOLLOW
                else hatching[row, column]
            )
        else:
            guide = tangent[row, column]
            if guide @ last_direction < 0:
                guide = -guide

            turn = CURVATURE * confidence
            direction = turn * guide + (1.0 - turn) * last_direction
            length = np.linalg.norm(direction)
            if length < 1e-9:
                break
            direction = direction / length

        position = position + radius * direction
        last_direction = direction
        points.append((float(position[0]), float(position[1])))

    return points


def _round_cap(
    centre: np.ndarray, direction: np.ndarray, normal: np.ndarray, half_width: float
) -> np.ndarray:
    """A half circle closing the end where the brush touched down.

    Swept in the local frame of the stroke: from one side, back around behind the
    starting point, to the other side. Appending it to the outline turns what would
    otherwise close as a straight line across the width into a rounded end.
    """
    angles = np.linspace(-np.pi / 2, np.pi / 2, CAP_POINTS)
    return centre + half_width * (
        np.cos(angles)[:, np.newaxis] * -direction
        + np.sin(angles)[:, np.newaxis] * normal
    )


def _bristle_tracks(
    points: list[tuple[float, float]], radius: float, generator
) -> list[list[tuple[float, float]]]:
    """Turn a path into a few tapered parallel ribbons, one per bristle.

    A constant-width band is the shape a mouse makes, not a brush. Three things separate
    the two, and all of them are about the ends:

    The brush lands, so the start is rounded rather than cut square. The brush lifts, so
    the mark narrows to almost nothing rather than stopping at full width. And the
    bristles are not all the same length, so they begin and end at slightly different
    points along the path instead of together on one line -- which is what stops a stroke
    looking as though it had been trimmed with scissors.
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

    offsets = (
        np.linspace(-1.0, 1.0, BRISTLES) if BRISTLES > 1 else np.zeros(1)
    ) * BRISTLE_SPREAD
    last_index = len(path) - 1

    tracks = []
    for offset in offsets:
        first = int(round(generator.uniform(0.0, BRISTLE_STAGGER) * last_index))
        final = int(round(generator.uniform(1.0 - BRISTLE_STAGGER, 1.0) * last_index))
        if final - first < 1:
            first, final = 0, last_index

        span = slice(first, final + 1)
        centre = path[span] + normals[span] * (
            offset * radius + generator.normal(0.0, 0.08 * radius)
        )
        edge = normals[span]

        along = np.linspace(0.0, 1.0, len(centre))
        taper = 1.0 - (1.0 - TAPER_TO) * along
        # Tapering and staggering between them remove about two fifths of the area a
        # square-ended track would cover, so the tracks are widened to put it back.
        # Without this the strokes are prettier and the painting is threadbare, with the
        # underpainting showing through everywhere.
        half_width = (
            (radius / BRISTLES) * 1.6 * generator.uniform(0.8, 1.2) * taper
        )

        left = centre + edge * half_width[:, np.newaxis]
        right = centre - edge * half_width[:, np.newaxis]
        cap = _round_cap(centre[0], directions[span][0], edge[0], half_width[0])

        outline = np.vstack([left, right[::-1], cap])
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
    jitter: float | None = None,
    video: str | None = None,
    fallback_angle: float = FALLBACK_ANGLE,
    tightness: float = DEFAULT_TIGHTNESS,
    separate: bool = False,
    background_tightness: float | None = None,
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

    style = Style.from_tightness(tightness)
    print(f"  tightness {tightness:.2f}: {style.describe()}")

    foreground, background_style = None, None
    if separate:
        foreground = foreground_mask(original)
        looser = (
            max(0.0, tightness - BACKGROUND_LOOSER_BY)
            if background_tightness is None
            else background_tightness
        )
        background_style = Style.from_tightness(looser)
        print(
            f"  subject covers {foreground.mean():.0%} of the picture; background "
            f"painted at tightness {looser:.2f}"
        )

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
        image,
        palette_colours,
        radii=radii,
        seed=seed,
        jitter=jitter,
        recorder=recorder,
        fallback_angle=fallback_angle,
        style=style,
        foreground=foreground,
        background_style=background_style,
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
        "-t",
        "--tightness",
        type=float,
        default=DEFAULT_TIGHTNESS,
        help="how closely the painting follows the photograph, from 0 for broad and "
        "loose to 1 for controlled and detailed (default %(default)s). Moves the brush "
        "sizes, the repaint threshold, the stroke length and the colour jitter together",
    )
    parser.add_argument(
        "--jitter",
        type=float,
        help="override the colour jitter the tightness would choose (0 to disable)",
    )
    parser.add_argument(
        "-s",
        "--separate",
        action="store_true",
        help="cut the subject out with rembg and paint the background first, then the "
        "subject over it, so the background can be looser and the silhouette stays clean",
    )
    parser.add_argument(
        "--background-tightness",
        type=float,
        help=f"tightness for the background (default: {BACKGROUND_LOOSER_BY} looser than "
        "the subject)",
    )
    parser.add_argument(
        "--fallback-angle",
        type=float,
        default=FALLBACK_ANGLE,
        help=f"degrees to hatch at where the picture has no direction of its own "
        f"(default %(default)s, drifting by up to {FALLBACK_DRIFT:.0f} across the canvas)",
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
        arguments.fallback_angle,
        arguments.tightness,
        arguments.separate,
        arguments.background_tightness,
    )
