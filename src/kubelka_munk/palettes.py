"""A ready-made palette, so the library can be tried in one line.

**These are not measured paints.** Every entry below is built with
:meth:`Paint.from_srgb`, from a hex colour eyeballed from paint-manufacturer colour
charts and a tinting strength set by judgement about how the real pigment behaves. The
names are those of real artists' colours because that is what makes the palette useful to
think with, but the numbers are approximations and should not be cited as pigment data.

No open dataset of Kubelka-Munk coefficients for artists' paints appears to exist -- the
values used by commercial colour-matching systems are proprietary. If you need accuracy,
measure your own paints and use :meth:`Paint.from_measurements`.

Tinting strength is the scattering coefficient, and sets how far a paint carries in a
mixture. Titanium white is far and away the strongest scatterer, which is why it
dominates; the transparent glazing colours scatter least.
"""

from __future__ import annotations

from .paint import Paint
from .palette import Palette

# name, masstone colour, tinting strength
_APPROXIMATE_PAINTS = [
    ("Titanium White", "#fbfaf6", 10.0),
    ("Ivory Black", "#23201e", 4.0),
    ("Cadmium Yellow Light", "#ffd300", 2.5),
    ("Yellow Ochre", "#c8a02c", 2.0),
    ("Cadmium Red Medium", "#e32227", 2.5),
    ("Alizarin Crimson", "#6f1a2e", 2.0),
    ("Quinacridone Magenta", "#8e2b5a", 3.0),
    ("Ultramarine Blue", "#2b3f9e", 1.5),
    ("Phthalo Blue", "#0d3b66", 6.0),
    ("Phthalo Green", "#123f2b", 6.0),
    ("Viridian", "#40826d", 1.2),
    ("Burnt Umber", "#4b3621", 1.5),
]

APPROXIMATE_ARTIST_PALETTE = Palette(
    Paint.from_srgb(name, colour, tinting_strength=strength)
    for name, colour, strength in _APPROXIMATE_PAINTS
)
