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
  4. Grow the stroke along the direction the image is *not* changing -- perpendicular to
     its gradient -- so strokes run along edges and around forms rather than across them.
     Stop when the stroke wanders somewhere it no longer belongs.

Stroke colours are snapped to mixtures of the four Zorn paints, so the result is a
picture that could in principle be painted with four tubes. Pass ``--mixtures N`` to
restrict it further to the N mixtures that suit this picture best.

Run it with:  python examples/zorn_painting.py <image> [output] [-m N] [--seed N]
"""

from __future__ import annotations

import argparse
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
MAXIMUM_STROKE = 16

# How readily a stroke changes direction: 1 follows the image gradient exactly and gives
# jittery strokes, 0 never turns at all. This is Hertzmann's curvature filter.
CURVATURE = 0.4

# How far a stroke's colour may wander from the palette mixture it started with before it
# is cut short, in units of colour difference.
STROKE_TOLERANCE = 25.0


def paint(
    image: np.ndarray,
    palette_colours: np.ndarray,
    radii: list[int] | None = None,
    seed: int = 0,
) -> Image.Image:
    """Work from the coarsest brush to the finest, refining what the last one missed."""
    radii = BRUSH_RADII if radii is None else radii
    height, width, _ = image.shape
    generator = np.random.default_rng(seed)

    snap = _palette_snapper(palette_colours)

    # Start from the picture's average colour, so that anywhere the brushes never visit
    # still reads as part of the painting rather than as a hole.
    canvas = Image.new("RGB", (width, height), tuple(snap(image.mean(axis=(0, 1)))))

    for radius in radii:
        reference = gaussian_filter(
            image, sigma=(BLUR_PER_RADIUS * radius, BLUR_PER_RADIUS * radius, 0)
        )
        strokes = _plan_layer(np.asarray(canvas) / 255.0, reference, radius)
        generator.shuffle(strokes)

        _draw_layer(canvas, strokes, reference, radius, snap)
        print(f"    brush {radius:3d}px  {len(strokes):6d} strokes")

    return canvas


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


def _draw_layer(canvas, strokes, reference, radius, snap) -> None:
    """Trace and paint every stroke of one layer."""
    brush = ImageDraw.Draw(canvas)

    # Strokes follow the image, so the direction to travel is worked out from the blurred
    # reference rather than from the painting, which is full of hard stroke edges.
    luminance = reference @ np.array([0.2126, 0.7152, 0.0722])
    smoothed = gaussian_filter(luminance, sigma=max(1.0, radius / 2))
    gradient_x = sobel(smoothed, axis=1)
    gradient_y = sobel(smoothed, axis=0)

    reference_lab = xyz_to_lab(srgb_to_xyz(reference))

    for x, y in strokes:
        colour = snap(reference[y, x])
        points = _trace_stroke(
            x, y, radius, gradient_x, gradient_y, reference_lab, reference_lab[y, x]
        )

        if len(points) > 1:
            brush.line(points, fill=colour, width=2 * radius, joint="curve")

        # Round caps, which a polyline alone does not give, and which are most of what
        # makes a mark read as a brush stroke rather than a ruled band.
        for cap_x, cap_y in (points[0], points[-1]):
            brush.ellipse(
                [cap_x - radius, cap_y - radius, cap_x + radius, cap_y + radius],
                fill=colour,
            )


def _trace_stroke(
    x: int,
    y: int,
    radius: int,
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
    reference_lab: np.ndarray,
    started_from: np.ndarray,
) -> list[tuple[int, int]]:
    """Follow the image from a starting point, turning as the picture turns.

    A stroke travels perpendicular to the gradient -- the direction in which the image is
    changing least -- so it runs along an edge instead of across it, which is what makes
    strokes seem to describe a form. It stops when it reaches somewhere whose colour no
    longer matches the colour the stroke is loaded with.
    """
    height, width = gradient_x.shape
    points = [(x, y)]
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

        gradient = np.array([gradient_x[row, column], gradient_y[row, column]])
        magnitude = np.linalg.norm(gradient)
        if magnitude < 1e-6:
            break

        # Perpendicular to the gradient, kept pointing the way we were already going so
        # the stroke does not double back on itself.
        direction = np.array([-gradient[1], gradient[0]]) / magnitude
        if direction @ last_direction < 0:
            direction = -direction

        if step > 0:
            direction = CURVATURE * direction + (1 - CURVATURE) * last_direction
            length = np.linalg.norm(direction)
            if length < 1e-6:
                break
            direction = direction / length

        position = position + radius * direction
        last_direction = direction
        points.append((int(round(position[0])), int(round(position[1]))))

    return points


def _palette_snapper(palette_colours: np.ndarray):
    """A function from any colour to the nearest mixture, as 8-bit RGB for drawing."""
    tree = cKDTree(xyz_to_lab(srgb_to_xyz(palette_colours)))
    as_bytes = np.clip(np.round(palette_colours * 255), 0, 255).astype(int)

    def snap(colour: np.ndarray) -> tuple[int, int, int]:
        _, index = tree.query(xyz_to_lab(srgb_to_xyz(np.clip(colour, 0.0, 1.0))))
        return tuple(as_bytes[index].tolist())

    return snap


def main(
    source: str,
    destination: str | None = None,
    mixtures: int | None = None,
    seed: int = 0,
    longest_side: int = 1400,
    radii: list[int] | None = None,
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

    painting = paint(image, palette_colours, radii=radii, seed=seed)
    painting.save(destination_path)

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
    )
