"""Building paints, from colours and from measurements."""

import numpy as np
import pytest

from kubelka_munk import Paint, Spectrum


def test_a_paint_built_from_a_colour_shows_that_colour():
    paint = Paint.from_srgb("Ultramarine Blue", "#2b3f9e")
    assert paint.hex_colour == "#2b3f9e"


@pytest.mark.parametrize(
    "colour", ["#000000", "#ffffff", "#0d3b66", "#ffd300", "#4b3621"]
)
def test_masstone_survives_the_trip_through_absorption_and_scattering(colour):
    """A paint's stated colour must come back out of the equations unchanged.

    This exercises the full chain -- reflectance recovery, the Saunderson inverse, the
    remission function, the split into K and S, and all of it run backwards. Dark
    colours are included deliberately: they are where an unwanted surface-reflection
    floor would show up.
    """
    assert Paint.from_srgb("test", colour).hex_colour == colour


def test_paints_from_colours_are_flagged_as_approximate():
    assert Paint.from_srgb("guessed", "#6b8e23").is_approximate


def test_tinting_strength_scales_scattering_without_changing_the_colour():
    weak = Paint.from_srgb("weak", "#8e2b5a", tinting_strength=0.5)
    strong = Paint.from_srgb("strong", "#8e2b5a", tinting_strength=5.0)

    assert weak.hex_colour == strong.hex_colour
    assert np.allclose(strong.scattering / weak.scattering, 10.0)
    # K/S is what fixes the colour, so it must be untouched.
    assert np.allclose(weak.absorption / weak.scattering, strong.absorption / strong.scattering)


def test_negative_coefficients_are_refused():
    with pytest.raises(ValueError, match="cannot be negative"):
        Paint("bad", absorption=-np.ones(36), scattering=np.ones(36))


def test_a_paint_that_scatters_nothing_is_refused():
    with pytest.raises(ValueError, match="invisible"):
        Paint("void", absorption=np.ones(36), scattering=np.zeros(36))


def test_wrongly_sized_coefficients_are_refused():
    with pytest.raises(ValueError, match="needs 36 values"):
        Paint("short", absorption=np.ones(5), scattering=np.ones(5))


def test_zero_tinting_strength_is_refused():
    with pytest.raises(ValueError, match="must be positive"):
        Paint.from_srgb("nothing", "#ffffff", tinting_strength=0.0)


class TestCalibrationFromMeasurements:
    """The two-constant calibration, checked by recovering coefficients we chose."""

    def test_measured_calibration_recovers_the_original_paint(self):
        """Mix a known paint with a known white, then calibrate from the result.

        If the calibration is right, feeding it a masstone and a tint generated from a
        paint whose K and S we already know must give those same K and S back.
        """
        from kubelka_munk import Palette

        white = Paint.from_srgb("White", "#fbfaf6", tinting_strength=10.0)
        original = Paint.from_srgb("Crimson", "#8e2b5a", tinting_strength=2.0)

        tint_fraction = 0.2
        tint_spectrum = Palette([original, white]).mix(
            [tint_fraction, 1 - tint_fraction]
        ).spectrum

        calibrated = Paint.from_measurements(
            "Crimson",
            masstone=original.spectrum,
            tint=tint_spectrum,
            white=white,
            tint_fraction=tint_fraction,
        )

        assert np.allclose(calibrated.scattering, original.scattering, rtol=1e-6)
        assert np.allclose(calibrated.absorption, original.absorption, rtol=1e-6)
        assert not calibrated.is_approximate

    def test_a_tint_fraction_outside_the_open_unit_interval_is_refused(self):
        white = Paint.from_srgb("White", "#ffffff", tinting_strength=10.0)
        spectrum = Spectrum.uniform(0.5)
        for fraction in (0.0, 1.0, -0.1, 1.5):
            with pytest.raises(ValueError, match="strictly between 0 and 1"):
                Paint.from_measurements(
                    "x", spectrum, spectrum, white, tint_fraction=fraction
                )

    def test_a_tint_identical_to_the_masstone_carries_no_information(self):
        """If white does not change the colour, nothing can be learned about scattering."""
        white = Paint.from_srgb("White", "#ffffff", tinting_strength=10.0)
        spectrum = Spectrum.uniform(0.4)
        with pytest.raises(ValueError, match="could not separate"):
            Paint.from_measurements(
                "flat", spectrum, spectrum, white, tint_fraction=0.5
            )
