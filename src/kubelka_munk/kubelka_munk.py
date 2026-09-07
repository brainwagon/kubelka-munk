"""The Kubelka-Munk equations themselves.

Kubelka and Munk (1931) modelled a paint film as a turbid medium in which light is
absorbed at a rate K and scattered back at a rate S per unit depth. For a film thick
enough that the surface underneath cannot be seen -- the only case this library models,
see docs/adr/0007 -- the whole theory reduces to the two functions below, which convert
between reflectance and the ratio K/S.

Everything here works on whole spectra at once: K, S and reflectance are all arrays with
one entry per wavelength band, and the equations apply band by band.
"""

from __future__ import annotations

import numpy as np

from .spectrum import MAXIMUM_REFLECTANCE, MINIMUM_REFLECTANCE

# Saunderson's two constants, k1 and k2, for a typical paint film.
#
# k1 defaults to zero, which deserves an explanation. It is the fraction of light
# reflected straight off the film's surface, around 0.04 for a paint with a gloss. But
# that light is what a spectrophotometer discards when it measures with the specular
# component excluded, which is the usual geometry for paint, and it is likewise absent
# from a colour named as an sRGB value, which describes how a surface looks and not what
# is glinting off it. Setting k1 to 0.04 alongside such data puts the surface reflection
# in twice, and imposes a floor: nothing could then appear darker than 4% reflectance,
# so every dark paint would come back a washed-out grey. Pass k1=0.04 when your
# measurements do include the specular component.
#
# k2, the light reflected back down from inside the film, has no such ambiguity and is
# always present. Published values vary between 0.4 and 0.6; 0.6 is the common choice.
DEFAULT_SURFACE_REFLECTION = 0.0
DEFAULT_INTERNAL_REFLECTION = 0.6


def absorption_over_scattering(reflectance: np.ndarray) -> np.ndarray:
    """K/S from the reflectance of an opaque film -- the Kubelka-Munk remission function.

        K/S = (1 - R)^2 / 2R

    The ratio runs from 0 for a perfect white to infinity for a perfect black, which is
    why reflectance is held strictly above zero.
    """
    reflectance = np.clip(
        np.asarray(reflectance, dtype=float), MINIMUM_REFLECTANCE, MAXIMUM_REFLECTANCE
    )
    return (1.0 - reflectance) ** 2 / (2.0 * reflectance)


def opaque_reflectance(absorption_over_scattering_ratio: np.ndarray) -> np.ndarray:
    """Reflectance of an infinitely thick film, inverting the remission function.

        R = 1 + K/S - sqrt((K/S)^2 + 2 K/S)

    This is the physically meaningful root of the quadratic; the other gives R > 1.
    """
    ratio = np.maximum(np.asarray(absorption_over_scattering_ratio, dtype=float), 0.0)
    reflectance = 1.0 + ratio - np.sqrt(ratio * ratio + 2.0 * ratio)
    return np.clip(reflectance, MINIMUM_REFLECTANCE, MAXIMUM_REFLECTANCE)


def apply_saunderson(
    internal_reflectance: np.ndarray,
    surface_reflection: float = DEFAULT_SURFACE_REFLECTION,
    internal_reflection: float = DEFAULT_INTERNAL_REFLECTION,
) -> np.ndarray:
    """Convert the reflectance Kubelka-Munk describes into what you would measure.

        R_measured = k1 + (1 - k1)(1 - k2) R / (1 - k2 R)

    Kubelka-Munk assumes light enters the film freely, but a real paint surface reflects
    some light straight back off the top without it ever meeting a pigment (k1), and
    reflects some back down again from the inside (k2). Ignoring this makes dark mixtures
    come out too light.
    """
    internal_reflectance = np.asarray(internal_reflectance, dtype=float)
    measured = surface_reflection + (
        (1.0 - surface_reflection)
        * (1.0 - internal_reflection)
        * internal_reflectance
        / (1.0 - internal_reflection * internal_reflectance)
    )
    return np.clip(measured, MINIMUM_REFLECTANCE, MAXIMUM_REFLECTANCE)


def remove_saunderson(
    measured_reflectance: np.ndarray,
    surface_reflection: float = DEFAULT_SURFACE_REFLECTION,
    internal_reflection: float = DEFAULT_INTERNAL_REFLECTION,
) -> np.ndarray:
    """Recover the internal reflectance from a measurement -- the inverse of ``apply_saunderson``.

        R = (R_measured - k1) / ((1 - k1)(1 - k2) + k2 (R_measured - k1))

    Apply this on the way in, before computing K/S from a measured or reconstructed
    reflectance, and ``apply_saunderson`` on the way back out. Applying either twice, or
    in the wrong order, is the mistake this pair exists to make hard.
    """
    measured_reflectance = np.asarray(measured_reflectance, dtype=float)
    above_surface = measured_reflectance - surface_reflection
    internal = above_surface / (
        (1.0 - surface_reflection) * (1.0 - internal_reflection)
        + internal_reflection * above_surface
    )
    return np.clip(internal, MINIMUM_REFLECTANCE, MAXIMUM_REFLECTANCE)
