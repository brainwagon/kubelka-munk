"""A colour wheel built from the Zorn palette by repeated halving.

Anders Zorn is said to have painted with four colours: yellow ochre, vermilion, ivory
black and white. This example takes the three chromatic ones, spaces them equally around
a circle, and then fills the circle in by repeatedly placing a new chit between each
neighbouring pair, mixed from them in equal proportion.

What makes this worth looking at is that the arcs are not gradients. Every chit is a
Kubelka-Munk mixture of real pigment coefficients, so the wheel bends where paint bends:
ivory black and yellow ochre meet in olive green, not in grey, because black is not a
neutral darkener but a pigment with a colour of its own.

Run it with:  python examples/zorn_wheel.py [rounds]
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass

import numpy as np

from kubelka_munk import Paint, Palette

# name, masstone, tinting strength -- approximations, as everywhere in this library.
ZORN_TRIAD = [
    ("Yellow Ochre", "#c8a02c", 2.0),
    ("Vermilion", "#e34234", 2.5),
    ("Ivory Black", "#23201e", 4.0),
]

SVG_FILENAME = "zorn_wheel.svg"


@dataclass(frozen=True)
class Chit:
    """One painted square on the wheel: where it sits, and what is in it.

    The weights are proportions of the three primaries. Holding them, rather than just a
    colour, is what lets two chits be mixed again: under Kubelka-Munk the coefficients of
    a mixture add in proportion, so mixing two chits in equal parts is exactly averaging
    their weights.
    """

    angle_degrees: float
    weights: np.ndarray
    generation: int


def build_wheel(palette: Palette, rounds: int) -> list[Chit]:
    """Space the primaries around a circle, then halve the gaps ``rounds`` times."""
    chits = [
        Chit(
            angle_degrees=index * 360.0 / len(palette),
            weights=np.eye(len(palette))[index],
            generation=0,
        )
        for index in range(len(palette))
    ]

    for generation in range(1, rounds + 1):
        chits = _fill_the_gaps(chits, generation)
    return chits


def _fill_the_gaps(chits: list[Chit], generation: int) -> list[Chit]:
    """Put a new chit between every neighbouring pair, mixed from them half and half."""
    filled: list[Chit] = []
    for position, chit in enumerate(chits):
        neighbour = chits[(position + 1) % len(chits)]

        # Going the short way round, so the last pair wraps past 360 correctly.
        gap = (neighbour.angle_degrees - chit.angle_degrees) % 360.0

        filled.append(chit)
        filled.append(
            Chit(
                angle_degrees=(chit.angle_degrees + gap / 2) % 360.0,
                weights=(chit.weights + neighbour.weights) / 2,
                generation=generation,
            )
        )
    return filled


def describe(palette: Palette, chit: Chit) -> tuple[str, str]:
    """The chit's colour, and its recipe written the way you would mix it."""
    mixture = palette.mix(chit.weights)
    recipe = ", ".join(
        f"{weight:.0%} {name}" for name, weight in mixture.parts(0.005).items()
    )
    return mixture.hex_colour, recipe


def print_the_ring(palette: Palette, chits: list[Chit], radius: int = 9) -> None:
    """Draw the wheel with coloured blocks, for terminals that can manage 24-bit colour."""
    height, width = 2 * radius + 3, 4 * radius + 6
    canvas = [[None] * width for _ in range(height)]

    for chit in chits:
        # Zero degrees at the top, running clockwise, as a colour wheel is usually drawn.
        radians = math.radians(chit.angle_degrees)
        row = round(radius + 1 - radius * math.cos(radians))
        column = round(width / 2 + 2 * radius * math.sin(radians))
        colour = palette.mix(chit.weights).hex_colour

        for cell in range(-2, 2):
            if 0 <= row < height and 0 <= column + cell < width:
                canvas[row][column + cell] = colour

    print()
    for line in canvas:
        print("".join(" " if c is None else _block(c) for c in line))


def _block(hex_colour: str) -> str:
    """One space, painted with an ANSI true-colour background."""
    red, green, blue = (int(hex_colour[i : i + 2], 16) for i in (1, 3, 5))
    return f"\033[48;2;{red};{green};{blue}m \033[0m"


def print_the_legend(palette: Palette, chits: list[Chit]) -> None:
    print("\n  angle   colour     mixed from")
    print("  " + "-" * 62)
    for chit in sorted(chits, key=lambda c: c.angle_degrees):
        colour, recipe = describe(palette, chit)
        marker = "*" if chit.generation == 0 else " "
        print(f"  {chit.angle_degrees:5.1f}  {marker} {_block(colour) * 3} {colour}  {recipe}")
    print("\n  * the three primaries; everything else is mixed from its neighbours")


def write_svg(palette: Palette, chits: list[Chit], path: str) -> None:
    """Write the wheel out as an SVG, since a colour wheel deserves better than a terminal."""
    size, ring_radius, chit_radius = 700, 210, 34
    centre = size / 2

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}">',
        f'<rect width="{size}" height="{size}" fill="#f4f2ee"/>',
        f'<circle cx="{centre}" cy="{centre}" r="{ring_radius}" fill="none" '
        f'stroke="#d8d4cc" stroke-width="1"/>',
    ]

    for chit in chits:
        radians = math.radians(chit.angle_degrees)
        x = centre + ring_radius * math.sin(radians)
        y = centre - ring_radius * math.cos(radians)
        colour, _ = describe(palette, chit)

        outline = "#3a3a3a" if chit.generation == 0 else "#b9b4ab"
        width = 2.5 if chit.generation == 0 else 1
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{chit_radius}" fill="{colour}" '
            f'stroke="{outline}" stroke-width="{width}"/>'
        )

        label_radius = ring_radius + chit_radius + 26
        parts.append(
            f'<text x="{centre + label_radius * math.sin(radians):.1f}" '
            f'y="{centre - label_radius * math.cos(radians):.1f}" '
            f'font-family="ui-monospace, monospace" font-size="11" fill="#55524c" '
            f'text-anchor="middle" dominant-baseline="middle">{colour}</text>'
        )

    parts.append(
        f'<text x="{centre}" y="{centre - 8}" font-family="Georgia, serif" '
        f'font-size="17" fill="#55524c" text-anchor="middle">Zorn palette</text>'
    )
    parts.append(
        f'<text x="{centre}" y="{centre + 14}" font-family="Georgia, serif" '
        f'font-size="12" fill="#8a857c" text-anchor="middle">'
        f'mixed with Kubelka-Munk</text>'
    )
    parts.append("</svg>")

    with open(path, "w") as handle:
        handle.write("\n".join(parts))


def main(rounds: int = 2) -> None:
    palette = Palette(
        Paint.from_srgb(name, colour, tinting_strength=strength)
        for name, colour, strength in ZORN_TRIAD
    )

    chits = build_wheel(palette, rounds)
    print(
        f"\n  {len(palette)} primaries, halved {rounds} time"
        f"{'s' if rounds != 1 else ''} -> {len(chits)} chits"
    )

    print_the_ring(palette, chits)
    print_the_legend(palette, chits)

    write_svg(palette, chits, SVG_FILENAME)
    print(f"\n  wrote {SVG_FILENAME}\n")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
