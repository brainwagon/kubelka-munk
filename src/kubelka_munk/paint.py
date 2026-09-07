"""A Paint: a named colorant described by how it absorbs and scatters light."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .colorimetry import spectrum_to_lab, spectrum_to_srgb, srgb_to_hex
from .kubelka_munk import (
    DEFAULT_INTERNAL_REFLECTION,
    DEFAULT_SURFACE_REFLECTION,
    absorption_over_scattering,
    apply_saunderson,
    opaque_reflectance,
    remove_saunderson,
)
from .spectrum import BAND_COUNT, Spectrum
from .upsampling import reflectance_from_srgb


@dataclass(frozen=True)
class Paint:
    """A colorant as it comes out of the tube.

    A Paint carries an absorption spectrum ``K`` and a scattering spectrum ``S``, both
    per unit concentration, plus the two constants describing how its dried surface
    reflects light. Mixing works by adding these spectra in proportion, which is the
    whole of Kubelka-Munk theory.

    Build one with :meth:`from_measurements` if you have measured reflectance data, or
    with :meth:`from_srgb` if all you have is a colour -- but read that method's warning.
    """

    name: str
    absorption: np.ndarray
    scattering: np.ndarray
    surface_reflection: float = DEFAULT_SURFACE_REFLECTION
    internal_reflection: float = DEFAULT_INTERNAL_REFLECTION
    is_approximate: bool = field(default=False)

    def __post_init__(self) -> None:
        for label, values in (
            ("absorption", self.absorption),
            ("scattering", self.scattering),
        ):
            array = np.asarray(values, dtype=float)
            if array.shape != (BAND_COUNT,):
                raise ValueError(
                    f"{self.name}: {label} needs {BAND_COUNT} values, "
                    f"got shape {array.shape}"
                )
            if np.any(array < 0.0):
                raise ValueError(
                    f"{self.name}: {label} cannot be negative -- a pigment cannot "
                    f"absorb or scatter a negative amount of light"
                )
            object.__setattr__(self, label, array)

        if np.all(self.scattering == 0.0):
            raise ValueError(f"{self.name}: a paint that scatters no light is invisible")

    @classmethod
    def from_srgb(
        cls,
        name: str,
        colour: str,
        tinting_strength: float = 1.0,
        surface_reflection: float = DEFAULT_SURFACE_REFLECTION,
        internal_reflection: float = DEFAULT_INTERNAL_REFLECTION,
    ) -> "Paint":
        """Build a Paint from its masstone colour alone. **This is an approximation.**

        A single colour does not determine how a paint behaves in a mixture. Two paints
        can look identical straight from the tube and behave completely differently once
        white is added, because that depends on how much they scatter, which their
        colour does not tell you. This constructor therefore *invents* the missing
        information, in two steps:

        1. A plausible reflectance curve is recovered from the colour (see
           :mod:`kubelka_munk.upsampling`). Many curves would produce this colour; the
           smoothest is chosen.
        2. That gives the ratio K/S, which is split into K and S by *assuming* the
           scattering is the same at every wavelength and equal to ``tinting_strength``.

        ``tinting_strength`` is the one dial over the guess, and it is worth setting.
        Raise it for an opaque, chalky paint that holds its own against white (a
        cadmium, an oxide); lower it, to perhaps 0.1, for a transparent glazing colour
        that a little white will overwhelm (a quinacridone, a phthalo). Leave it at 1.0
        and every paint tints alike, which no real palette does.

        Paints made this way are marked ``is_approximate``. For real work, measure your
        paints and use :meth:`from_measurements`.
        """
        if tinting_strength <= 0.0:
            raise ValueError(f"{name}: tinting strength must be positive")

        measured = reflectance_from_srgb(colour).reflectance
        internal = remove_saunderson(measured, surface_reflection, internal_reflection)

        scattering = np.full(BAND_COUNT, float(tinting_strength))
        absorption = absorption_over_scattering(internal) * scattering
        return cls(
            name=name,
            absorption=absorption,
            scattering=scattering,
            surface_reflection=surface_reflection,
            internal_reflection=internal_reflection,
            is_approximate=True,
        )

    @classmethod
    def from_measurements(
        cls,
        name: str,
        masstone: Spectrum,
        tint: Spectrum,
        white: "Paint",
        tint_fraction: float,
        surface_reflection: float = DEFAULT_SURFACE_REFLECTION,
        internal_reflection: float = DEFAULT_INTERNAL_REFLECTION,
    ) -> "Paint":
        """Build a Paint from measured reflectance -- the honest way.

        This is the standard two-constant calibration. Absorption and scattering cannot
        be separated from one measurement, because only their ratio affects the colour
        of an opaque film. Adding a known white in a known proportion breaks the tie:
        the white contributes scattering you already know, so the amount the tint
        lightens reveals how much the paint itself scatters.

        Args:
            masstone: reflectance of the paint applied thickly and unmixed.
            tint: reflectance of the paint mixed with ``white``.
            white: the white paint used for the tint, itself already calibrated.
            tint_fraction: the proportion of *this* paint in the tint, in (0, 1).
                A 1:9 paint-to-white tint means 0.1.

        Scattering is recovered per wavelength and clamped at zero, since measurement
        noise can otherwise drive it slightly negative, which is unphysical.
        """
        if not 0.0 < tint_fraction < 1.0:
            raise ValueError(
                f"{name}: tint fraction must lie strictly between 0 and 1, "
                f"got {tint_fraction}"
            )

        masstone_ratio = absorption_over_scattering(
            remove_saunderson(
                masstone.reflectance, surface_reflection, internal_reflection
            )
        )
        tint_ratio = absorption_over_scattering(
            remove_saunderson(tint.reflectance, surface_reflection, internal_reflection)
        )

        # In the tint, K and S are both concentration-weighted sums of this paint's and
        # the white's. Writing c for the tint fraction and r for the tint's K/S:
        #
        #     (c K + (1-c) K_w) = r (c S + (1-c) S_w)
        #
        # and the masstone gives K = m S for this paint's own ratio m. Substituting and
        # solving for S leaves one linear equation per wavelength.
        white_fraction = 1.0 - tint_fraction
        numerator = white_fraction * (
            tint_ratio * white.scattering - white.absorption
        )
        denominator = tint_fraction * (masstone_ratio - tint_ratio)

        scattering = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=np.abs(denominator) > 1e-12,
        )
        scattering = np.maximum(scattering, 0.0)
        if np.all(scattering == 0.0):
            raise ValueError(
                f"{name}: could not separate absorption from scattering -- the tint is "
                f"indistinguishable from the masstone, so the measurements carry no "
                f"information about scattering"
            )

        return cls(
            name=name,
            absorption=masstone_ratio * scattering,
            scattering=scattering,
            surface_reflection=surface_reflection,
            internal_reflection=internal_reflection,
            is_approximate=False,
        )

    @property
    def spectrum(self) -> Spectrum:
        """The reflectance of this paint applied thickly and unmixed -- its masstone."""
        internal = opaque_reflectance(self.absorption / self.scattering)
        return Spectrum(
            apply_saunderson(
                internal, self.surface_reflection, self.internal_reflection
            )
        )

    @property
    def srgb(self) -> np.ndarray:
        """The masstone colour as sRGB in [0, 1]."""
        return spectrum_to_srgb(self.spectrum)

    @property
    def hex_colour(self) -> str:
        """The masstone colour as ``#rrggbb``."""
        return srgb_to_hex(self.srgb)

    @property
    def lab(self) -> np.ndarray:
        """The masstone colour in CIELAB."""
        return spectrum_to_lab(self.spectrum)

    def __repr__(self) -> str:
        marker = " (approximate)" if self.is_approximate else ""
        return f"<Paint {self.name!r} {self.hex_colour}{marker}>"
