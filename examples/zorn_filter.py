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

Run it with:  python examples/zorn_filter.py <image> [output]
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from kubelka_munk import srgb_to_xyz, xyz_to_lab
from zorn_wheel import add_tints, build_hues, build_palette, ZORN_HUES

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
    hues = build_hues(len(ZORN_HUES), HUE_ROUNDS)
    chits = add_tints(hues, TINT_STEPS, most_white=MOST_WHITE)

    weights = [chit.weights for chit in chits]

    # Plain white, which no tinted hue quite reaches, for the highlights.
    pure_white = np.zeros(len(palette))
    pure_white[-1] = 1.0
    weights.append(pure_white)

    colours = np.array([palette.mix(weight).srgb for weight in weights])
    return colours, weights


def repaint(image: np.ndarray, chit_colours: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Replace every pixel with the nearest mixture, and say how far it had to move.

    Nearest is measured in CIELAB, where distance is roughly perceived difference, rather
    than in sRGB, where it is not. The search is a k-d tree over the mixtures: a few
    hundred candidates against a few million pixels is far too much work to do by brute
    force, and the tree makes it a lookup.
    """
    pixels_lab = xyz_to_lab(srgb_to_xyz(image)).reshape(-1, 3)
    chits_lab = xyz_to_lab(srgb_to_xyz(chit_colours))

    distances, chosen = cKDTree(chits_lab).query(pixels_lab, workers=-1)

    height, width, _ = image.shape
    return chosen.reshape(height, width), distances.reshape(height, width)


def report(palette, weights, chosen: np.ndarray, distances: np.ndarray) -> None:
    """Which paints the picture actually needed, and where the palette ran out."""
    counts = Counter(chosen.ravel().tolist())
    total = chosen.size

    print(f"\n  {len(weights)} mixtures available, {len(counts)} of them used\n")
    print("  share   colour     mixed from")
    print("  " + "-" * 66)
    for index, count in counts.most_common(8):
        mixture = palette.mix(weights[index])
        recipe = ", ".join(
            f"{weight:.0%} {name}" for name, weight in mixture.parts(0.02).items()
        )
        print(f"  {count / total:5.1%}   {mixture.hex_colour}   {recipe}")

    print(f"\n  colour difference from the original, over the whole image:")
    print(f"    median {np.median(distances):5.2f}    mean {distances.mean():5.2f}"
          f"    worst {distances.max():6.2f}")
    unreachable = float((distances > 10).mean())
    print(
        f"    {unreachable:.1%} of pixels landed further than 10 away -- these are the "
        f"blues and\n    greens the palette has no answer for."
    )


def main(source: str, destination: str | None = None) -> None:
    source_path = Path(source)
    destination_path = (
        Path(destination)
        if destination
        else source_path.with_name(source_path.stem + "_zorn.png")
    )

    image = np.asarray(Image.open(source_path).convert("RGB"), dtype=float) / 255.0
    print(f"\n  {source_path.name}: {image.shape[1]} x {image.shape[0]}")

    palette = build_palette()
    chit_colours, weights = build_chit_colours(palette)

    chosen, distances = repaint(image, chit_colours)
    repainted = np.clip(np.round(chit_colours[chosen] * 255), 0, 255).astype(np.uint8)

    Image.fromarray(repainted).save(destination_path)
    report(palette, weights, chosen, distances)
    print(f"\n  wrote {destination_path}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python examples/zorn_filter.py <image> [output]")
    main(*sys.argv[1:3])
