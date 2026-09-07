"""Recovering a plausible reflectance curve from a single sRGB colour.

Countless different reflectance curves look like the same colour, so this problem has no
one right answer. Burns's method picks the *smoothest* curve that reproduces the target
colour exactly, on the grounds that real paints reflect light smoothly across the
spectrum -- no natural pigment has a spiky reflectance.

The variant implemented here is least hyperbolic-tangent slope squared (LHTSS). Rather
than solving for reflectance directly, it solves for an unbounded curve z and defines

    R = (1 + tanh(z)) / 2

so reflectance is confined to (0, 1) by construction rather than by clamping, which is
what the plain least-slope-squared variant needs and often violates.

    Burns, "Generating Reflectance Curves from sRGB Triplets", arXiv:1710.05732
    https://arxiv.org/pdf/1710.05732
    Burns, "Numerical Methods for Smoothest Reflectance Reconstruction" (2020)
    http://scottburns.us/wp-content/uploads/2025/03/Numerical-Methods-for-Smoothest-Reflectance-Reconstruction-Burns-2020.pdf
"""

from __future__ import annotations

import numpy as np

from .colorimetry import (
    REFLECTANCE_TO_XYZ,
    XYZ_TO_LINEAR_SRGB,
    decode_srgb,
    hex_to_srgb,
)
from .spectrum import BAND_COUNT, Spectrum

# Reflectance in, linear sRGB out: integrate against illuminant and observer, then rotate
# into the display's primaries. This is the constraint the recovered curve must satisfy.
_REFLECTANCE_TO_LINEAR_SRGB = REFLECTANCE_TO_XYZ @ XYZ_TO_LINEAR_SRGB.T

# Roughness is measured as the sum of squared differences between neighbouring bands.
# _ROUGHNESS is the matrix that turns a curve into that sum: z @ _ROUGHNESS @ z.
_FIRST_DIFFERENCES = np.diff(np.eye(BAND_COUNT), axis=0)
_ROUGHNESS = _FIRST_DIFFERENCES.T @ _FIRST_DIFFERENCES

_MAXIMUM_ITERATIONS = 100
_CONVERGENCE_TOLERANCE = 1e-12

# Reflectance is (1 + tanh(z)) / 2, which approaches 0 and 1 only as z runs off to
# infinity. A pure black or pure white target is therefore unreachable exactly, so
# targets are held a hair inside the achievable range.
#
# The two margins differ because the two ends do not behave alike. Near black the
# recovered curve stays well conditioned, so the margin can be tiny and pure black comes
# back as pure black. Near white the curve must saturate at every wavelength at once,
# which needs more room before the solve stops being degenerate. Both offsets are far
# below one step of an 8-bit channel.
_DARK_MARGIN = 1e-4
_LIGHT_MARGIN = 1e-3


def reflectance_from_srgb(colour: str | np.ndarray) -> Spectrum:
    """The smoothest reflectance curve that reproduces this sRGB colour under D65.

    Accepts a hex string such as ``"#1f3a93"`` or an array of three values in [0, 1].

    The result is *a* spectrum that looks like the given colour, not the spectrum of any
    particular real paint. See docs/adr/0008.
    """
    srgb = hex_to_srgb(colour) if isinstance(colour, str) else np.asarray(colour, float)
    if srgb.shape != (3,):
        raise ValueError(f"expected three sRGB channels, got shape {srgb.shape}")
    if np.any(srgb < 0.0) or np.any(srgb > 1.0):
        raise ValueError(f"sRGB channels must lie in [0, 1], got {srgb}")

    target_linear_srgb = np.clip(
        decode_srgb(srgb), _DARK_MARGIN, 1.0 - _LIGHT_MARGIN
    )
    return Spectrum(_smoothest_curve_matching(target_linear_srgb))


def _smoothest_curve_matching(target_linear_srgb: np.ndarray) -> np.ndarray:
    """Minimise curve roughness subject to hitting the target colour exactly.

    A constrained minimisation solved by Newton's method on the Lagrangian. The unknowns
    are the curve ``z`` (one value per band) and three Lagrange multipliers, one per
    colour channel; the solution is where the gradient of the Lagrangian vanishes.
    """
    curve = np.zeros(BAND_COUNT)  # tanh(0) = 0, so we start from mid grey.
    multipliers = np.zeros(3)

    for _ in range(_MAXIMUM_ITERATIONS):
        reflectance = (1.0 + np.tanh(curve)) / 2.0
        # d(reflectance)/d(curve) and its second derivative.
        slope = (1.0 - np.tanh(curve) ** 2) / 2.0
        curvature = -np.tanh(curve) * (1.0 - np.tanh(curve) ** 2)

        constraint_jacobian = _REFLECTANCE_TO_LINEAR_SRGB.T * slope  # (3, bands)

        roughness_gradient = 2.0 * _ROUGHNESS @ curve
        gradient = np.concatenate(
            [
                roughness_gradient + constraint_jacobian.T @ multipliers,
                _REFLECTANCE_TO_LINEAR_SRGB.T @ reflectance - target_linear_srgb,
            ]
        )
        if np.max(np.abs(gradient)) < _CONVERGENCE_TOLERANCE:
            break

        constraint_curvature = np.diag(
            (_REFLECTANCE_TO_LINEAR_SRGB @ multipliers) * curvature
        )
        hessian = np.block(
            [
                [2.0 * _ROUGHNESS + constraint_curvature, constraint_jacobian.T],
                [constraint_jacobian, np.zeros((3, 3))],
            ]
        )
        # Least squares rather than a direct solve: where the curve saturates, tanh's
        # slope vanishes and the system becomes singular. Least squares takes the
        # smallest step satisfying it, which is well defined either way.
        step, *_ = np.linalg.lstsq(hessian, -gradient, rcond=None)
        curve += step[:BAND_COUNT]
        multipliers += step[BAND_COUNT:]
    else:
        raise RuntimeError(
            "reflectance recovery did not converge; the requested colour may lie "
            "outside the sRGB gamut"
        )

    return (1.0 + np.tanh(curve)) / 2.0
