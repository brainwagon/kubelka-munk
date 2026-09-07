"""Colorimetry: the standard conversions, checked against published values."""

from pathlib import Path

import numpy as np
import pytest

from kubelka_munk import Spectrum, delta_e_1976, delta_e_2000
from kubelka_munk.colorimetry import (
    D65_WHITE_POINT_XYZ,
    decode_srgb,
    encode_srgb,
    hex_to_srgb,
    spectrum_to_srgb,
    spectrum_to_xyz,
    srgb_to_hex,
    srgb_to_lab,
    srgb_to_xyz,
    xyz_to_lab,
    xyz_to_srgb,
)

CIEDE2000_TEST_DATA = Path(__file__).parent / "ciede2000_test_data.txt"


def test_perfect_white_reflector_has_the_d65_white_point():
    """A surface reflecting everything must take the colour of the light falling on it."""
    white = spectrum_to_xyz(Spectrum.uniform(1.0))

    # The nominal D65 white point is (0.9505, 1.0000, 1.0890). Ours is computed from the
    # tables decimated to 10 nm, so it agrees to about three decimal places, not exactly.
    # Reflectance is held a whisker below 1 to keep the Kubelka-Munk equations finite,
    # so Y lands just under 1 rather than exactly on it.
    assert white[1] == pytest.approx(1.0, abs=1e-8)
    assert white[0] == pytest.approx(0.9505, abs=1e-3)
    assert white[2] == pytest.approx(1.0890, abs=1e-3)


def test_perfect_white_reflector_is_white_on_screen():
    assert srgb_to_hex(spectrum_to_srgb(Spectrum.uniform(1.0))) == "#ffffff"


def test_white_point_has_no_colour_cast_in_lab():
    lightness, green_red, blue_yellow = xyz_to_lab(D65_WHITE_POINT_XYZ)
    assert lightness == pytest.approx(100.0)
    assert green_red == pytest.approx(0.0, abs=1e-9)
    assert blue_yellow == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("value", [0.0, 0.001, 0.0031308, 0.5, 0.9, 1.0])
def test_srgb_transfer_function_inverts_itself(value):
    assert decode_srgb(encode_srgb(np.array([value])))[0] == pytest.approx(value)


@pytest.mark.parametrize(
    "colour", ["#000000", "#ffffff", "#1f3a93", "#ff0000", "#7f7f7f"]
)
def test_hex_survives_a_round_trip_through_xyz(colour):
    assert srgb_to_hex(xyz_to_srgb(srgb_to_xyz(hex_to_srgb(colour)))) == colour


def test_short_hex_notation_expands():
    assert np.allclose(hex_to_srgb("#f0a"), hex_to_srgb("#ff00aa"))


@pytest.mark.parametrize("bad", ["#12345", "not a colour", "#gggggg", ""])
def test_malformed_hex_is_rejected(bad):
    with pytest.raises(ValueError):
        hex_to_srgb(bad)


def test_delta_e_is_zero_for_identical_colours():
    lab = srgb_to_lab(hex_to_srgb("#6b8e23"))
    assert delta_e_1976(lab, lab) == pytest.approx(0.0)
    assert delta_e_2000(lab, lab) == pytest.approx(0.0)


def _sharma_pairs():
    """The 34 test pairs published with the CIEDE2000 implementation notes."""
    for line in CIEDE2000_TEST_DATA.read_text().splitlines():
        numbers = [float(field) for field in line.split()]
        if len(numbers) == 7:
            yield numbers[0:3], numbers[3:6], numbers[6]


@pytest.mark.parametrize("first, second, expected", list(_sharma_pairs()))
def test_ciede2000_matches_the_sharma_test_data(first, second, expected):
    """Pinned against Sharma, Wu and Dalal (2005), whose data file ships in this directory.

    These pairs are chosen to exercise every awkward corner of the formula -- the hue
    discontinuity at 360 degrees, the blue-region rotation term, near-neutral colours.
    Passing them is the accepted proof that an implementation is correct.
    """
    assert delta_e_2000(first, second) == pytest.approx(expected, abs=1e-4)


def test_ciede2000_is_symmetric():
    first = srgb_to_lab(hex_to_srgb("#1f3a93"))
    second = srgb_to_lab(hex_to_srgb("#8e2b5a"))
    assert delta_e_2000(first, second) == pytest.approx(delta_e_2000(second, first))


@pytest.mark.parametrize(
    "colour", ["#000000", "#ffffff", "#1f3a93", "#6b8e23", "#7f7f7f"]
)
def test_lab_converts_back_to_the_colour_it_came_from(colour):
    from kubelka_munk import lab_to_srgb

    assert srgb_to_hex(lab_to_srgb(srgb_to_lab(hex_to_srgb(colour)))) == colour


def test_colour_conversions_work_on_a_whole_stack_at_once():
    """An image is millions of colours; converting them one at a time is not an option.

    Every conversion takes the three components on the last axis, so a stack of any
    shape goes through in one call and must agree with the colours done singly.
    """
    from kubelka_munk import lab_to_xyz

    colours = np.array(
        [hex_to_srgb(c) for c in ("#000000", "#ffffff", "#1f3a93", "#6b8e23", "#c08040")]
    )

    stacked = xyz_to_lab(srgb_to_xyz(colours))
    assert stacked.shape == colours.shape
    for index, colour in enumerate(colours):
        assert stacked[index] == pytest.approx(xyz_to_lab(srgb_to_xyz(colour)))

    assert xyz_to_srgb(lab_to_xyz(stacked)) == pytest.approx(colours, abs=1e-9)

    # Shape is preserved, so an image-shaped array works without reshaping.
    image = colours.reshape(1, 5, 3)
    assert xyz_to_lab(srgb_to_xyz(image)).shape == (1, 5, 3)
