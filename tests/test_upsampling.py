"""Recovering reflectance curves from colours."""

import numpy as np
import pytest

from kubelka_munk import reflectance_from_srgb, spectrum_to_srgb, srgb_to_hex
from kubelka_munk.spectrum import BAND_COUNT

CORNERS_AND_PRIMARIES = [
    "#000000",
    "#ffffff",
    "#ff0000",
    "#00ff00",
    "#0000ff",
    "#ffff00",
    "#00ffff",
    "#ff00ff",
    "#7f7f7f",
    "#1f3a93",
]


@pytest.mark.parametrize("colour", CORNERS_AND_PRIMARIES)
def test_recovered_curve_reproduces_the_colour_it_came_from(colour):
    """The defining property: the curve must look like the colour that produced it."""
    assert srgb_to_hex(spectrum_to_srgb(reflectance_from_srgb(colour))) == colour


def test_every_colour_in_a_random_sweep_round_trips():
    """Includes the gamut corners, where the tanh parametrisation is hardest pressed."""
    generator = np.random.default_rng(20240607)
    for _ in range(200):
        channels = generator.integers(0, 256, 3)
        colour = "#{:02x}{:02x}{:02x}".format(*channels)
        assert srgb_to_hex(spectrum_to_srgb(reflectance_from_srgb(colour))) == colour


@pytest.mark.parametrize("colour", CORNERS_AND_PRIMARIES)
def test_recovered_reflectance_is_physically_possible(colour):
    """A surface cannot reflect less than none or more than all of the light."""
    reflectance = reflectance_from_srgb(colour).reflectance
    assert reflectance.shape == (BAND_COUNT,)
    assert np.all(reflectance > 0.0)
    assert np.all(reflectance < 1.0)


def test_recovered_curves_are_smooth():
    """Smoothness is the whole principle of the method, so it is worth asserting.

    Real pigments have no sharp spectral features at this resolution; a curve that
    jumped between neighbouring bands would be a sign the solver had gone wrong.
    """
    reflectance = reflectance_from_srgb("#6b8e23").reflectance
    assert np.max(np.abs(np.diff(reflectance))) < 0.1


def test_a_neutral_grey_recovers_a_flat_curve():
    reflectance = reflectance_from_srgb("#7f7f7f").reflectance
    assert np.ptp(reflectance) < 0.01


@pytest.mark.parametrize("bad", [[1.5, 0.0, 0.0], [-0.1, 0.0, 0.0]])
def test_out_of_range_channels_are_rejected(bad):
    with pytest.raises(ValueError):
        reflectance_from_srgb(np.array(bad))
