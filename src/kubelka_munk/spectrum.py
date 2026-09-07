"""The wavelength grid, and reflectance sampled on it."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FIRST_WAVELENGTH_NM = 380
LAST_WAVELENGTH_NM = 730
WAVELENGTH_STEP_NM = 10

WAVELENGTHS_NM = np.arange(
    FIRST_WAVELENGTH_NM, LAST_WAVELENGTH_NM + 1, WAVELENGTH_STEP_NM
)
BAND_COUNT = len(WAVELENGTHS_NM)

# Reflectance is held strictly inside (0, 1). Exactly zero makes the Kubelka-Munk
# remission function divide by zero, and exactly one leaves no room for the numerical
# slack that optimisation introduces. See docs/kubelka-munk-theory.md, "Numerical care".
MINIMUM_REFLECTANCE = 1e-6
MAXIMUM_REFLECTANCE = 1.0 - 1e-9


@dataclass(frozen=True)
class Spectrum:
    """Diffuse reflectance sampled at each wavelength in ``WAVELENGTHS_NM``.

    Values are fractions in (0, 1): 0.5 means half the incident light at that
    wavelength comes back. A Spectrum describes a *surface*, not a light source.
    """

    reflectance: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.reflectance, dtype=float)
        if values.shape != (BAND_COUNT,):
            raise ValueError(
                f"a Spectrum needs one reflectance per band: expected {BAND_COUNT} "
                f"values for {FIRST_WAVELENGTH_NM}-{LAST_WAVELENGTH_NM} nm in "
                f"{WAVELENGTH_STEP_NM} nm steps, got shape {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("reflectance contains NaN or infinity")
        clamped = np.clip(values, MINIMUM_REFLECTANCE, MAXIMUM_REFLECTANCE)
        object.__setattr__(self, "reflectance", clamped)

    @classmethod
    def uniform(cls, reflectance: float) -> "Spectrum":
        """A flat spectrum -- the same reflectance at every wavelength (a neutral grey)."""
        return cls(np.full(BAND_COUNT, float(reflectance)))

    def __len__(self) -> int:
        return BAND_COUNT
