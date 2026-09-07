"""The Kubelka-Munk equations and the Saunderson correction."""

import numpy as np
import pytest

from kubelka_munk import (
    absorption_over_scattering,
    apply_saunderson,
    opaque_reflectance,
    remove_saunderson,
)
from kubelka_munk.kubelka_munk import (
    DEFAULT_INTERNAL_REFLECTION,
    DEFAULT_SURFACE_REFLECTION,
)

REFLECTANCES = np.array([0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])


def test_a_perfect_white_absorbs_nothing():
    assert absorption_over_scattering(np.array([1.0]))[0] == pytest.approx(0.0)


def test_darker_surfaces_have_a_higher_absorption_ratio():
    """K/S is the quantity that rises as a colour deepens -- it is why the model works."""
    ratios = absorption_over_scattering(REFLECTANCES)
    assert np.all(np.diff(ratios) < 0)  # reflectance ascending, so K/S descending


def test_remission_function_inverts_itself():
    recovered = opaque_reflectance(absorption_over_scattering(REFLECTANCES))
    assert recovered == pytest.approx(REFLECTANCES, abs=1e-12)


def test_a_totally_black_surface_does_not_produce_infinity():
    """K/S diverges as reflectance goes to zero; the clamp has to catch it."""
    ratio = absorption_over_scattering(np.array([0.0]))
    assert np.isfinite(ratio).all()
    assert opaque_reflectance(ratio)[0] < 1e-3


def test_saunderson_inverts_itself():
    recovered = remove_saunderson(apply_saunderson(REFLECTANCES, 0.04, 0.6), 0.04, 0.6)
    assert recovered == pytest.approx(REFLECTANCES, abs=1e-12)


def test_saunderson_with_zero_constants_changes_nothing():
    assert apply_saunderson(REFLECTANCES, 0.0, 0.0) == pytest.approx(REFLECTANCES)
    assert remove_saunderson(REFLECTANCES, 0.0, 0.0) == pytest.approx(REFLECTANCES)


def test_saunderson_darkens_mid_tones():
    """Internal reflection is what stops the model predicting washed-out darks."""
    measured = apply_saunderson(REFLECTANCES, 0.0, DEFAULT_INTERNAL_REFLECTION)
    assert np.all(measured < REFLECTANCES)


def test_surface_reflection_sets_a_floor_on_how_dark_a_film_can_look():
    """With a specular component included, nothing can appear darker than the surface.

    This is the reason the library defaults k1 to zero: sRGB colours and
    specular-excluded measurements have no such floor, and imposing one turns every dark
    paint grey.
    """
    measured = apply_saunderson(np.array([0.0]), 0.04, 0.6)
    assert measured[0] == pytest.approx(0.04)


def test_defaults_are_the_documented_ones():
    assert DEFAULT_SURFACE_REFLECTION == 0.0
    assert DEFAULT_INTERNAL_REFLECTION == 0.6
