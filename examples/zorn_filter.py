"""Repaint a photograph in the Zorn palette.

Builds the same wheel of mixtures as ``zorn_wheel.py`` -- hues by repeated halving,
tinted with white -- and then replaces every pixel of an image with whichever mixture
comes closest to it. The output is a picture that could, in principle, be painted with
four tubes of paint.

The interesting part is what the palette cannot do. Yellow ochre, vermilion, ivory black
and white span the warm half of colour space and nothing else, which is exactly why Zorn
used them for portraits: skin lives there. Anything blue or green in the photograph has
nowhere to go and lands on the nearest warm grey. The report printed at the end says how
far off each region ended up, so the failures are measured rather than guessed at.

Two options are worth knowing about.

``--mixtures N`` cuts the palette down to the N mixtures that between them cover this
particular picture best, which is the practically interesting question: not which colours
sit evenly around a wheel, but which ones you would actually squeeze out to paint this.

``--dither`` spreads each pixel's rounding error into its neighbours instead of throwing
it away. It costs a little graininess and buys back the tones the palette does not have,
which matters most exactly when the palette is small. At twelve mixtures the difference
between the two is the difference between a posterised cutout and a painting.

Run it with:  python examples/zorn_filter.py <image> [output] [-m N] [--dither]
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from kubelka_munk import decode_srgb, srgb_to_xyz, xyz_to_lab
from kubelka_munk.colorimetry import LINEAR_SRGB_TO_XYZ
from zorn_wheel import add_tints, build_hues, build_palette, PRUSSIAN_BLUE

# A denser wheel than the one drawn on screen: an image needs somewhere to put every
# pixel, so it wants more hues and, especially, more steps of value -- skin is mostly a
# gradient of lightness, and too coarse a ladder shows up as blotches across a cheek.
#
# Past this there is little to gain. Doubling both again, to 769 mixtures, moves the
# median colour difference only from 3.95 to 3.80: what is left is not the palette being
# too coarse but the palette being the wrong shape, having no blue in it at all.
HUE_ROUNDS = 3
TINT_STEPS = 16

# Tinted further than the drawn wheel goes, because photographs have highlights. Stopping
# short of 1.0 keeps the palest tints distinct from each other -- at 1.0 every hue would
# collapse onto plain white, which is added once, separately.
MOST_WHITE = 0.94


def build_chit_colours(palette) -> tuple[np.ndarray, list[np.ndarray]]:
    """Every mixture the palette offers, as sRGB, alongside the recipe that makes it.

    Returns the colours as an (n, 3) array and the matching weight vectors, so a pixel's
    nearest colour can be traced back to the paint that produced it.
    """
    # Every paint on the palette but the white, which is the tinting axis rather than a
    # point on the wheel.
    hues = build_hues(len(palette) - 1, HUE_ROUNDS)
    chits = add_tints(hues, TINT_STEPS, most_white=MOST_WHITE)

    weights = [chit.weights for chit in chits]

    # Plain white, which no tinted hue quite reaches, for the highlights.
    pure_white = np.zeros(len(palette))
    pure_white[-1] = 1.0
    weights.append(pure_white)

    colours = np.array([palette.mix(weight).srgb for weight in weights])
    return colours, weights


def choose_best_mixtures(
    image: np.ndarray, chit_colours: np.ndarray, wanted: int, seed: int = 0
) -> np.ndarray:
    """Pick the handful of mixtures that between them cover this picture best.

    Which twelve mixtures should go on the palette for *this* painting is a different
    question from which twelve are evenly spread around the wheel, and the answer depends
    on the picture: a portrait wants a dozen skin tones and one grey, a seascape the
    reverse.

    This is a greedy cover. Start with the single mixture that suits the image best, then
    repeatedly add whichever remaining mixture most reduces the total error, counting each
    pixel only against its own nearest choice. Greedy is not optimal -- the optimal subset
    is a hard combinatorial problem -- but it is within a whisker for this kind of job and
    takes a second rather than a week.

    Returns indices into ``chit_colours``.
    """
    if wanted >= len(chit_colours):
        return np.arange(len(chit_colours))

    # A sample is plenty: which mixtures are worth having is a question about the broad
    # distribution of colour in the image, not about individual pixels.
    pixels = image.reshape(-1, 3)
    generator = np.random.default_rng(seed)
    sample = pixels[
        generator.choice(len(pixels), size=min(20000, len(pixels)), replace=False)
    ]

    sample_lab = xyz_to_lab(srgb_to_xyz(sample))
    chits_lab = xyz_to_lab(srgb_to_xyz(chit_colours))

    # Every pixel against every candidate, once. Everything below is bookkeeping on this.
    errors = np.linalg.norm(sample_lab[:, None, :] - chits_lab[None, :, :], axis=2)

    chosen: list[int] = []
    best_so_far = np.full(len(sample_lab), np.inf)
    for _ in range(wanted):
        # If this candidate joined the palette, how much error would be left?
        remaining = np.minimum(best_so_far[:, None], errors).sum(axis=0)
        remaining[chosen] = np.inf

        winner = int(np.argmin(remaining))
        chosen.append(winner)
        best_so_far = np.minimum(best_so_far, errors[:, winner])

    return np.array(sorted(chosen))


def repaint(image: np.ndarray, chit_colours: np.ndarray) -> np.ndarray:
    """Replace every pixel with the nearest mixture.

    Nearest is measured in CIELAB, where distance is roughly perceived difference, rather
    than in sRGB, where it is not. The search is a k-d tree over the mixtures: a few
    hundred candidates against a few million pixels is far too much work to do by brute
    force, and the tree makes it a lookup.
    """
    pixels_lab = xyz_to_lab(srgb_to_xyz(image)).reshape(-1, 3)
    chits_lab = xyz_to_lab(srgb_to_xyz(chit_colours))

    _, chosen = cKDTree(chits_lab).query(pixels_lab, workers=-1)

    height, width, _ = image.shape
    return chosen.reshape(height, width)


def repaint_with_dithering(image: np.ndarray, chit_colours: np.ndarray) -> np.ndarray:
    """Repaint, spreading each pixel's rounding error into its neighbours.

    Plain nearest-neighbour has no way to express a colour between two mixtures, so a
    smooth gradient across a cheek breaks into flat patches. Floyd-Steinberg error
    diffusion buys the missing tones back: whatever a pixel could not represent is
    handed on to the pixels not yet visited, which then land differently, and the average
    over a small area comes out right. It is what a painter does by scumbling one colour
    thinly over another.

    The error is carried in linear light rather than in display values, because that is
    what physically averages when two small patches of paint are seen from across a room.

    Ordinarily this is strictly serial -- each pixel depends on its left neighbour, so
    there is nothing to vectorise and three million Python iterations to do. But the
    dependencies all point the same way: a pixel takes error only from (x-1, y),
    (x+1, y-1), (x, y-1) and (x-1, y-1), and every one of those has a smaller value of
    x + 2y. So all the pixels along a line of constant x + 2y are independent of each
    other and can be quantised together, which turns three million steps into about five
    thousand.
    """
    height, width, _ = image.shape
    chits_lab = xyz_to_lab(srgb_to_xyz(chit_colours))
    chits_linear = decode_srgb(chit_colours)
    tree = cKDTree(chits_lab)

    working = decode_srgb(image).copy()
    chosen = np.zeros((height, width), dtype=np.intp)

    rows, columns = np.indices((height, width))
    wavefront_of = (columns + 2 * rows).ravel()
    order = np.argsort(wavefront_of, kind="stable")
    boundaries = np.searchsorted(wavefront_of[order], np.arange(wavefront_of.max() + 2))

    flat_rows, flat_columns = rows.ravel()[order], columns.ravel()[order]

    # Floyd-Steinberg: where each sixteenth of the error goes.
    spread = [(0, 1, 7 / 16), (1, -1, 3 / 16), (1, 0, 5 / 16), (1, 1, 1 / 16)]

    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        if start == stop:
            continue
        y, x = flat_rows[start:stop], flat_columns[start:stop]

        # Clamped, and this is load-bearing rather than defensive. The palette has no
        # blue in it, so a blue pixel leaves an error pointing in a direction nothing can
        # ever correct; passed on undiminished it accumulates, and by the far corner of
        # the picture the carried error is a hundred times larger than any real colour.
        # A pixel has received all its error by the time its own wavefront comes round,
        # so clamping here loses nothing that was going to arrive later -- it just
        # declines to chase colours that do not exist.
        wanted = np.clip(working[y, x], 0.0, 1.0)

        _, pick = tree.query(xyz_to_lab(wanted @ LINEAR_SRGB_TO_XYZ.T))
        chosen[y, x] = pick

        error = wanted - chits_linear[pick]
        for down, across, share in spread:
            target_y, target_x = y + down, x + across
            inside = (
                (target_y < height) & (target_x >= 0) & (target_x < width)
            )
            # add.at, not +=, because two pixels on one wavefront can feed the same
            # neighbour, and plain indexed assignment would keep only one of them.
            np.add.at(
                working,
                (target_y[inside], target_x[inside]),
                error[inside] * share,
            )

    return chosen


def measure(image: np.ndarray, chit_colours: np.ndarray, chosen: np.ndarray) -> np.ndarray:
    """How far each pixel ended up from the colour it started as.

    Always measured against the original photograph, never against the error-adjusted
    value the dithering was working with -- otherwise dithering would be marking its own
    homework, and would score well precisely where it had strayed furthest.
    """
    original_lab = xyz_to_lab(srgb_to_xyz(image))
    painted_lab = xyz_to_lab(srgb_to_xyz(chit_colours[chosen]))
    return np.linalg.norm(original_lab - painted_lab, axis=-1)


def report(
    palette, weights, chosen: np.ndarray, distances: np.ndarray, listed: int = 8
) -> None:
    """Which paints the picture actually needed, and where the palette ran out."""
    counts = Counter(chosen.ravel().tolist())
    total = chosen.size

    print(f"\n  {len(weights)} mixtures available, {len(counts)} of them used\n")
    print("  share   colour     mixed from")
    print("  " + "-" * 66)
    for index, count in counts.most_common(listed):
        mixture = palette.mix(weights[index])
        recipe = ", ".join(
            f"{weight:.0%} {name}" for name, weight in mixture.parts(0.02).items()
        )
        print(f"  {count / total:5.1%}   {mixture.hex_colour}   {recipe}")

    print(f"\n  colour difference from the original, pixel by pixel:")
    print(f"    median {np.median(distances):5.2f}    mean {distances.mean():5.2f}"
          f"    worst {distances.max():6.2f}")
    unreachable = float((distances > 10).mean())
    print(
        f"    {unreachable:.1%} of pixels landed further than 10 away -- these are the "
        f"blues and\n    greens the palette has no answer for."
    )


def main(
    source: str,
    destination: str | None = None,
    mixtures: int | None = None,
    dither: bool = False,
    blue: bool = False,
) -> None:
    source_path = Path(source)
    destination_path = (
        Path(destination)
        if destination
        else source_path.with_name(source_path.stem + "_zorn.png")
    )

    image = np.asarray(Image.open(source_path).convert("RGB"), dtype=float) / 255.0
    print(f"\n  {source_path.name}: {image.shape[1]} x {image.shape[0]}")

    palette = build_palette([PRUSSIAN_BLUE] if blue else [])
    chit_colours, weights = build_chit_colours(palette)

    if mixtures is not None:
        keep = choose_best_mixtures(image, chit_colours, mixtures)
        chit_colours = chit_colours[keep]
        weights = [weights[index] for index in keep]
        print(f"  chose the {len(keep)} mixtures that suit this picture best")

    repainter = repaint_with_dithering if dither else repaint
    print(f"  mapping{' with dithering' if dither else ''}...")
    chosen = repainter(image, chit_colours)
    distances = measure(image, chit_colours, chosen)

    repainted = np.clip(np.round(chit_colours[chosen] * 255), 0, 255).astype(np.uint8)
    Image.fromarray(repainted).save(destination_path)

    report(palette, weights, chosen, distances, listed=min(len(weights), 12))
    print(f"\n  wrote {destination_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", help="the image to repaint")
    parser.add_argument(
        "destination", nargs="?", help="where to write it (default: alongside the source)"
    )
    parser.add_argument(
        "-m",
        "--mixtures",
        type=int,
        help="use only this many mixtures, chosen greedily to suit this picture "
        "(default: every mixture on the wheel)",
    )
    parser.add_argument(
        "-b",
        "--blue",
        action="store_true",
        help="add Prussian blue to the palette, which Zorn's four have no answer for",
    )
    parser.add_argument(
        "-d",
        "--dither",
        action="store_true",
        help="spread each pixel's error into its neighbours, trading a little "
        "graininess for much smoother gradients",
    )
    arguments = parser.parse_args()
    main(
        arguments.source,
        arguments.destination,
        arguments.mixtures,
        arguments.dither,
        arguments.blue,
    )
