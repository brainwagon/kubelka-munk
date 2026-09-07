"""Mixing artists' paints with Kubelka-Munk theory.

Two things this library does:

    >>> from kubelka_munk import APPROXIMATE_ARTIST_PALETTE as palette
    >>> palette.mix({"Phthalo Blue": 1, "Cadmium Yellow Light": 1})   # what do I get?
    >>> palette.match("#6b8e23", max_paints=3)                        # how do I get it?

Mixing is spectral: each paint is described by how much light it absorbs and scatters at
each wavelength, those quantities add in proportion, and the colour is worked out from
the result. That is why blue and yellow give green here, and a plain blend of the two
colours gives grey.

Read docs/kubelka-munk-theory.md for the theory and the approximations, and CONTEXT.md
for what the words mean.
"""

from .colorimetry import (
    delta_e_1976,
    delta_e_2000,
    hex_to_srgb,
    lab_to_srgb,
    lab_to_xyz,
    spectrum_to_lab,
    spectrum_to_srgb,
    srgb_to_hex,
    srgb_to_lab,
    srgb_to_xyz,
    xyz_to_lab,
    xyz_to_srgb,
)
from .kubelka_munk import (
    absorption_over_scattering,
    apply_saunderson,
    opaque_reflectance,
    remove_saunderson,
)
from .paint import Paint
from .palette import Mixture, Palette, Recipe, as_spectrum, spectrum_from_lab
from .palettes import APPROXIMATE_ARTIST_PALETTE
from .spectrum import BAND_COUNT, WAVELENGTHS_NM, Spectrum
from .upsampling import reflectance_from_srgb

__all__ = [
    "APPROXIMATE_ARTIST_PALETTE",
    "BAND_COUNT",
    "Mixture",
    "Paint",
    "Palette",
    "Recipe",
    "Spectrum",
    "WAVELENGTHS_NM",
    "absorption_over_scattering",
    "apply_saunderson",
    "as_spectrum",
    "delta_e_1976",
    "delta_e_2000",
    "hex_to_srgb",
    "lab_to_srgb",
    "lab_to_xyz",
    "opaque_reflectance",
    "reflectance_from_srgb",
    "remove_saunderson",
    "spectrum_from_lab",
    "spectrum_to_lab",
    "spectrum_to_srgb",
    "srgb_to_hex",
    "srgb_to_lab",
    "srgb_to_xyz",
    "xyz_to_lab",
    "xyz_to_srgb",
]
