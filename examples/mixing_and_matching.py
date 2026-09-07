"""A tour of the library: mix some paints, then work backwards from a colour.

Run it with:  python examples/mixing_and_matching.py
"""

from kubelka_munk import APPROXIMATE_ARTIST_PALETTE as palette
from kubelka_munk import Palette, hex_to_srgb, srgb_to_hex


def show_the_palette() -> None:
    print("The palette (approximate -- see palettes.py):\n")
    for paint in palette:
        print(f"  {paint.hex_colour}  {paint.name}")


def mix_blue_and_yellow() -> None:
    """The demonstration that pigment mixing is not colour blending."""
    print("\n\nUltramarine to cadmium yellow, mixed as paint and as pixels:\n")
    pair = Palette([palette["Ultramarine Blue"], palette["Cadmium Yellow Light"]])
    blue = hex_to_srgb(palette["Ultramarine Blue"].hex_colour)
    yellow = hex_to_srgb(palette["Cadmium Yellow Light"].hex_colour)

    print("  yellow    as paint   as pixels")
    for fraction in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0):
        as_paint = pair.mix([1 - fraction, fraction]).hex_colour
        as_pixels = srgb_to_hex(blue * (1 - fraction) + yellow * fraction)
        print(f"   {fraction:4.0%}     {as_paint}    {as_pixels}")
    print("\n  The paint column runs through green. The pixel column runs through mud.")


def match_some_colours() -> None:
    print("\n\nWorking backwards -- what do I mix to get this colour?\n")
    for target in ("#6b8e23", "#8a9a5b", "#c08040", "#404080", "#00ff00"):
        recipe = palette.match(target, max_paints=3)
        parts = ", ".join(
            f"{name} {weight:.0%}" for name, weight in recipe.parts(1e-2).items()
        )
        verdict = "good match" if recipe.is_good_match else "not quite reachable"
        print(f"  {target} -> {recipe.mixture.hex_colour}  dE {recipe.delta_e:5.2f}  {verdict}")
        print(f"            {parts}\n")

    print("  Pure display green is beyond any pigment, and the library says so rather")
    print("  than quietly returning its best guess as though it had succeeded.")


def rank_the_alternatives() -> None:
    print("\nSeveral ways to reach one colour, closest first:\n")
    for recipe in palette.match_all("#8a9a5b", max_paints=2, limit=4):
        parts = ", ".join(
            f"{name} {weight:.0%}" for name, weight in recipe.parts(1e-2).items()
        )
        print(f"  dE {recipe.delta_e:5.2f}   {parts}")


if __name__ == "__main__":
    show_the_palette()
    mix_blue_and_yellow()
    match_some_colours()
    rank_the_alternatives()
