"""A colour wheel built from the Zorn palette by repeated halving, then tinted with white.

Anders Zorn is said to have painted with four colours: yellow ochre, vermilion, ivory
black and white. This example takes the three chromatic ones, spaces them equally around
a circle, and fills the circle in by repeatedly placing a new chit between each
neighbouring pair, mixed from them in equal proportion. That gives the hues.

Titanium white is then added to every one of those chits in increasing proportion, and
each tint is placed further out from the centre. So angle carries the hue and radius
carries the white: the middle of the disc is the pure mixtures, the rim is those same
mixtures at their palest.

The tints are the part worth looking at. Ivory black is so much the strongest absorber
here that most of the wheel collapses into near-black before any white is added -- three
parts vermilion to one part black is already almost unreadable. Add white and those same
mixtures open out into the greens, olives and mauves that were in them all along.

Run it with:  python examples/zorn_wheel.py [rounds] [tints]
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass

import numpy as np

from kubelka_munk import Paint, Palette, reflectance_from_srgb

# Each paint is given two observations rather than one: its masstone, and its colour
# when one part of it is mixed into nine parts titanium white. The tint is what separates
# absorption from scattering, and it is the difference between a wheel that works and one
# whose black end is a smear of near-identical browns.
#
# A masstone alone cannot do this. It fixes only the ratio K/S, and every paint sharing
# that ratio -- from one that barely tints to one that annihilates everything it touches
# -- has the identical masstone. The tint is the observation that tells them apart, and
# it is something a painter can judge by eye far more reliably than a coefficient.
#
# These are estimates, not measurements. Replace them with your own paints' tints.
TINT_FRACTION = 0.1

# name, masstone, colour at one part in nine of white
ZORN_HUES = [
    ("Yellow Ochre", "#c8a02c", "#e5d7b0"),
    ("Vermilion", "#e34234", "#f3c0b0"),
    ("Ivory Black", "#23201e", "#9e9e9d"),
]

# The white everything else is calibrated against, and the only paint still built from a
# single colour -- there is nothing lighter to tint it with.
TITANIUM_WHITE = ("Titanium White", "#fbfaf6", 10.0)

# How pale the outermost ring gets. Titanium white scatters so strongly that going much
# beyond this washes every hue out to the same near-white.
MOST_WHITE = 0.75

SVG_FILENAME = "zorn_wheel.svg"


@dataclass(frozen=True)
class Chit:
    """One painted square on the disc: where it sits, and what is in it.

    The weights are proportions of the whole palette, white included. Holding them,
    rather than just a colour, is what lets two chits be mixed again: under Kubelka-Munk
    the coefficients of a mixture add in proportion, so mixing two chits in equal parts
    is exactly averaging their weights.
    """

    angle_degrees: float
    ring: int
    white_fraction: float
    weights: np.ndarray
    is_primary: bool


def build_hues(hue_count: int, rounds: int) -> list[tuple[float, np.ndarray]]:
    """Space the primaries around a circle, then halve the gaps ``rounds`` times.

    Returns angles paired with proportions of the chromatic paints only. White is added
    afterwards, so that halving mixes hue with hue and never with a tint.
    """
    hues = [
        (index * 360.0 / hue_count, np.eye(hue_count)[index]) for index in range(hue_count)
    ]
    for _ in range(rounds):
        hues = _fill_the_gaps(hues)
    return hues


def _fill_the_gaps(
    hues: list[tuple[float, np.ndarray]]
) -> list[tuple[float, np.ndarray]]:
    """Put a new hue between every neighbouring pair, mixed from them half and half."""
    filled: list[tuple[float, np.ndarray]] = []
    for position, (angle, weights) in enumerate(hues):
        neighbour_angle, neighbour_weights = hues[(position + 1) % len(hues)]

        # Going the short way round, so the last pair wraps past 360 correctly.
        gap = (neighbour_angle - angle) % 360.0

        filled.append((angle, weights))
        filled.append(((angle + gap / 2) % 360.0, (weights + neighbour_weights) / 2))
    return filled


def add_tints(
    hues: list[tuple[float, np.ndarray]], tints: int, most_white: float = MOST_WHITE
) -> list[Chit]:
    """Give every hue a run of chits, each with more white than the last.

    Ring 0 holds the hue itself. Each ring outwards replaces a larger share of the chit
    with titanium white, so a chit's distance from the centre is how much white is in it.
    """
    white_fractions = (
        np.linspace(0.0, most_white, tints) if tints > 1 else np.zeros(1)
    )

    chits = []
    for angle, hue_weights in hues:
        for ring, white_fraction in enumerate(white_fractions):
            chits.append(
                Chit(
                    angle_degrees=angle,
                    ring=ring,
                    white_fraction=float(white_fraction),
                    weights=np.append(
                        hue_weights * (1.0 - white_fraction), white_fraction
                    ),
                    is_primary=bool(np.max(hue_weights) == 1.0),
                )
            )
    return chits


def describe(palette: Palette, chit: Chit) -> tuple[str, str]:
    """The chit's colour, and its recipe written the way you would mix it."""
    mixture = palette.mix(chit.weights)
    recipe = ", ".join(
        f"{weight:.0%} {name}" for name, weight in mixture.parts(0.005).items()
    )
    return mixture.hex_colour, recipe


def print_the_disc(palette: Palette, chits: list[Chit], rings: int) -> None:
    """Draw the disc with coloured blocks, for terminals that can manage 24-bit colour."""
    innermost, spacing = 6, 3
    outermost = innermost + spacing * (rings - 1)
    height, width = 2 * outermost + 3, 4 * outermost + 8
    canvas: list[list[str | None]] = [[None] * width for _ in range(height)]

    for chit in chits:
        # Zero degrees at the top, running clockwise, as a colour wheel is usually drawn.
        radians = math.radians(chit.angle_degrees)
        radius = innermost + spacing * chit.ring
        row = round(outermost + 1 - radius * math.cos(radians))
        column = round(width / 2 + 2 * radius * math.sin(radians))
        colour = palette.mix(chit.weights).hex_colour

        for cell in (-1, 0):
            if 0 <= row < height and 0 <= column + cell < width:
                canvas[row][column + cell] = colour

    print()
    for line in canvas:
        print("".join(" " if c is None else _block(c) for c in line))
    print("\n  centre: no white.  rim: {:.0%} white.".format(MOST_WHITE))


def _block(hex_colour: str) -> str:
    """One space, painted with an ANSI true-colour background."""
    red, green, blue = (int(hex_colour[i : i + 2], 16) for i in (1, 3, 5))
    return f"\033[48;2;{red};{green};{blue}m \033[0m"


def print_the_legend(palette: Palette, chits: list[Chit], rings: int) -> None:
    """One row per hue, one column per tint, so the tinting can be read across."""
    by_angle: dict[float, list[Chit]] = {}
    for chit in chits:
        by_angle.setdefault(chit.angle_degrees, []).append(chit)

    fractions = [chit.white_fraction for chit in sorted(by_angle[0.0], key=lambda c: c.ring)]
    header = "".join(f"  {fraction:>6.0%} white" for fraction in fractions)
    print(f"\n  angle {header}   hue")
    print("  " + "-" * (20 + 13 * rings))

    for angle in sorted(by_angle):
        row = sorted(by_angle[angle], key=lambda c: c.ring)
        cells = ""
        for chit in row:
            colour, _ = describe(palette, chit)
            cells += f"  {_block(colour) * 2} {colour}"

        _, hue_recipe = describe(palette, row[0])
        marker = "*" if row[0].is_primary else " "
        print(f"  {angle:5.1f} {marker}{cells}   {hue_recipe}")

    print("\n  * the three chromatic primaries; the rest are mixed from their neighbours")


def write_svg(
    palette: Palette,
    chits: list[Chit],
    rings: int,
    label_angle: float,
    path: str,
) -> None:
    """Write the disc out as an SVG, since a colour wheel deserves better than a terminal."""
    innermost, spacing, chit_radius = 96, 52, 21
    outermost = innermost + spacing * (rings - 1)
    size = 2 * (outermost + chit_radius + 34)
    centre = size / 2

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}">',
        f'<rect width="{size}" height="{size}" fill="#f4f2ee"/>',
    ]

    for ring in range(rings):
        parts.append(
            f'<circle cx="{centre}" cy="{centre}" r="{innermost + spacing * ring}" '
            f'fill="none" stroke="#e2ded6" stroke-width="1"/>'
        )

    for chit in chits:
        radians = math.radians(chit.angle_degrees)
        radius = innermost + spacing * chit.ring
        x = centre + radius * math.sin(radians)
        y = centre - radius * math.cos(radians)
        colour, _ = describe(palette, chit)

        # The undiluted primaries are ringed, so the three corners stay findable once
        # the disc fills up.
        ringed = chit.is_primary and chit.ring == 0
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{chit_radius}" fill="{colour}" '
            f'stroke="{"#3a3a3a" if ringed else "#cdc8bf"}" '
            f'stroke-width="{2.5 if ringed else 0.8}"/>'
        )

    # The ring labels go on a ray that falls between two hues, so they never land on a
    # chit however finely the wheel is subdivided.
    label_radians = math.radians(label_angle)
    for ring in range(rings):
        radius = innermost + spacing * ring
        x = centre + radius * math.sin(label_radians)
        y = centre - radius * math.cos(label_radians)
        # A pill in the background colour, so a label stays readable even where the ring
        # it names runs close to a chit.
        parts.append(
            f'<rect x="{x - 14:.1f}" y="{y - 8:.1f}" width="28" height="16" rx="8" '
            f'fill="#f4f2ee"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" '
            f'font-family="ui-monospace, monospace" font-size="10" fill="#9a958c" '
            f'text-anchor="middle" dominant-baseline="middle">'
            f'{chits[ring].white_fraction:.0%}</text>'
        )
    parts.append(
        f'<text x="{centre + (outermost + chit_radius + 16) * math.sin(label_radians):.1f}" '
        f'y="{centre - (outermost + chit_radius + 16) * math.cos(label_radians):.1f}" '
        f'font-family="ui-monospace, monospace" font-size="10" fill="#8a857c" '
        f'text-anchor="middle" dominant-baseline="middle">white</text>'
    )

    parts.append(
        f'<text x="{centre}" y="{centre - 6}" font-family="Georgia, serif" '
        f'font-size="16" fill="#55524c" text-anchor="middle">Zorn palette</text>'
    )
    parts.append(
        f'<text x="{centre}" y="{centre + 14}" font-family="Georgia, serif" '
        f'font-size="11" fill="#8a857c" text-anchor="middle">'
        f'hue around, white outward</text>'
    )
    parts.append("</svg>")

    with open(path, "w") as handle:
        handle.write("\n".join(parts))


def build_palette() -> Palette:
    """The four Zorn paints: three calibrated from masstone and tint, plus the white."""
    white = Paint.from_srgb(*TITANIUM_WHITE[:2], tinting_strength=TITANIUM_WHITE[2])

    hues = [
        Paint.from_measurements(
            name,
            masstone=reflectance_from_srgb(masstone),
            tint=reflectance_from_srgb(tint),
            white=white,
            tint_fraction=TINT_FRACTION,
        )
        for name, masstone, tint in ZORN_HUES
    ]
    return Palette([*hues, white])


def main(rounds: int = 2, tints: int = 4) -> None:
    palette = build_palette()

    hues = build_hues(len(ZORN_HUES), rounds)
    chits = add_tints(hues, tints)
    print(
        f"\n  {len(ZORN_HUES)} primaries, halved {rounds} time"
        f"{'s' if rounds != 1 else ''} -> {len(hues)} hues"
        f", each tinted {tints} ways -> {len(chits)} chits"
    )

    print_the_disc(palette, chits, tints)
    print_the_legend(palette, chits, tints)

    # A ray halfway between the first two hues, kept clear for the ring labels.
    label_angle = (hues[0][0] + hues[1][0]) / 2 if len(hues) > 1 else 45.0

    write_svg(palette, chits, tints, label_angle, SVG_FILENAME)
    print(f"\n  wrote {SVG_FILENAME}\n")


if __name__ == "__main__":
    arguments = [int(value) for value in sys.argv[1:3]]
    main(*arguments)
