"""Mixing paints and searching for recipes."""

import numpy as np
import pytest

from kubelka_munk import (
    APPROXIMATE_ARTIST_PALETTE,
    Paint,
    Palette,
    Spectrum,
    delta_e_2000,
    hex_to_srgb,
    srgb_to_hex,
    srgb_to_lab,
)


@pytest.fixture
def palette():
    return APPROXIMATE_ARTIST_PALETTE


class TestMixing:
    def test_mixing_a_paint_with_itself_returns_that_paint(self, palette):
        """The most basic sanity check there is: nothing happens."""
        for fraction in (0.1, 0.5, 0.9):
            pure = Palette([palette["Ultramarine Blue"]])
            mixture = pure.mix([fraction])
            assert mixture.hex_colour == palette["Ultramarine Blue"].hex_colour

    def test_all_the_weight_on_one_paint_gives_that_paint(self, palette):
        for index, paint in enumerate(palette):
            weights = np.zeros(len(palette))
            weights[index] = 1.0
            assert palette.mix(weights).hex_colour == paint.hex_colour

    def test_weights_are_normalised_so_parts_work_like_a_painter_expects(self, palette):
        """"Two parts blue to one part white" should not have to be written as fractions."""
        by_parts = palette.mix({"Ultramarine Blue": 2, "Titanium White": 1})
        by_fraction = palette.mix(
            {"Ultramarine Blue": 2 / 3, "Titanium White": 1 / 3}
        )
        assert by_parts.hex_colour == by_fraction.hex_colour
        assert by_parts.weights.sum() == pytest.approx(1.0)

    def test_scaling_all_the_weights_changes_nothing(self, palette):
        once = palette.mix({"Cadmium Red Medium": 1, "Titanium White": 3})
        ten_times = palette.mix({"Cadmium Red Medium": 10, "Titanium White": 30})
        assert once.hex_colour == ten_times.hex_colour

    def test_mixing_is_order_independent(self, palette):
        forwards = Palette(
            [palette["Ultramarine Blue"], palette["Cadmium Yellow Light"]]
        ).mix([0.3, 0.7])
        backwards = Palette(
            [palette["Cadmium Yellow Light"], palette["Ultramarine Blue"]]
        ).mix([0.7, 0.3])
        assert forwards.hex_colour == backwards.hex_colour

    def test_white_lightens(self, palette):
        red = palette["Cadmium Red Medium"]
        tinted = palette.mix({"Cadmium Red Medium": 1, "Titanium White": 3})
        # Cadmium red is a strong tinter, so three parts white lifts it by about 19
        # points of lightness rather than washing it out entirely.
        assert tinted.lab[0] > red.lab[0] + 15

    def test_black_darkens(self, palette):
        white = palette["Titanium White"]
        greyed = palette.mix({"Titanium White": 3, "Ivory Black": 1})
        assert greyed.lab[0] < white.lab[0] - 20

    def test_a_stronger_tinter_dominates_a_weaker_one(self):
        """Tinting strength must actually decide who wins a mixture."""
        weak = Paint.from_srgb("weak", "#0d3b66", tinting_strength=0.2)
        strong = Paint.from_srgb("strong", "#0d3b66", tinting_strength=20.0)
        white = Paint.from_srgb("white", "#ffffff", tinting_strength=10.0)

        weak_tint = Palette([weak, white]).mix([0.1, 0.9])
        strong_tint = Palette([strong, white]).mix([0.1, 0.9])
        assert strong_tint.lab[0] < weak_tint.lab[0]

    def test_negative_weights_are_refused(self, palette):
        with pytest.raises(ValueError, match="cannot be negative"):
            palette.mix({"Titanium White": -1, "Ivory Black": 2})

    def test_all_zero_weights_are_refused(self, palette):
        with pytest.raises(ValueError, match="nothing to mix"):
            palette.mix({"Titanium White": 0})

    def test_an_unknown_paint_name_is_refused(self, palette):
        with pytest.raises(KeyError):
            palette.mix({"Unobtainium Violet": 1})

    def test_the_wrong_number_of_weights_is_refused(self, palette):
        with pytest.raises(ValueError, match="one weight per paint"):
            palette.mix([0.5, 0.5])


class TestBlueAndYellowMakeGreen:
    """The behaviour that justifies the whole library.

    Pigments mix subtractively: blue absorbs the long wavelengths, yellow absorbs the
    short ones, and green is what neither of them takes away. Averaging the two colours
    directly cannot produce this -- it runs through grey, which is why digital painting
    without a pigment model looks wrong to painters.
    """

    @staticmethod
    def _is_green(lab):
        lightness, green_red, blue_yellow = lab
        return green_red < -5 and blue_yellow > 5

    def test_blue_and_yellow_mix_to_green(self, palette):
        mixture = palette.mix(
            {"Ultramarine Blue": 1, "Cadmium Yellow Light": 3}
        )
        assert self._is_green(mixture.lab), f"got {mixture.hex_colour}"

    def test_averaging_the_two_colours_directly_does_not_give_green(self, palette):
        """The control: this is what the library exists to avoid."""
        blue = hex_to_srgb(palette["Ultramarine Blue"].hex_colour)
        yellow = hex_to_srgb(palette["Cadmium Yellow Light"].hex_colour)
        blended_lab = srgb_to_lab(0.25 * blue + 0.75 * yellow)
        assert not self._is_green(blended_lab)

    def test_the_path_from_blue_to_yellow_passes_through_green(self, palette):
        """Not just one mixing ratio -- the whole transition should be green."""
        pair = Palette(
            [palette["Ultramarine Blue"], palette["Cadmium Yellow Light"]]
        )
        greens = [
            self._is_green(pair.mix([1 - f, f]).lab) for f in np.linspace(0.5, 0.95, 10)
        ]
        assert sum(greens) >= 5


class TestMatching:
    def test_a_palette_colour_is_matched_exactly(self, palette):
        """Any paint on the palette is trivially reachable -- use all of it."""
        recipe = palette.match(palette["Viridian"].spectrum)
        assert recipe.delta_e < 0.5
        assert recipe.parts()["Viridian"] > 0.95

    def test_a_known_mixture_is_recovered(self, palette):
        """Mix something, hide the recipe, and see whether the search finds it again."""
        pair = Palette(
            [palette["Cadmium Red Medium"], palette["Titanium White"]]
        )
        target = pair.mix([0.3, 0.7])

        recipe = pair.match(target.spectrum)
        assert recipe.delta_e < 1.0
        assert recipe.weights[0] == pytest.approx(0.3, abs=0.05)

    def test_recipe_weights_are_a_valid_set_of_proportions(self, palette):
        recipe = palette.match("#6b8e23", max_paints=3)
        assert recipe.weights.sum() == pytest.approx(1.0)
        assert np.all(recipe.weights >= 0.0)

    def test_max_paints_is_respected(self, palette):
        recipe = palette.match("#8a9a5b", max_paints=3)
        assert len(recipe.parts(minimum_weight=1e-3)) <= 3

    def test_a_target_can_be_given_as_hex_or_srgb_or_spectrum(self, palette):
        by_hex = palette.match("#8a9a5b", max_paints=2)
        by_srgb = palette.match(hex_to_srgb("#8a9a5b"), max_paints=2)
        assert by_hex.mixture.hex_colour == by_srgb.mixture.hex_colour

    def test_an_unreachable_target_reports_its_distance_instead_of_failing(self, palette):
        """A pure display green is far outside what any pigment can do."""
        recipe = palette.match("#00ff00")
        assert recipe.delta_e > 10
        assert not recipe.is_good_match

    def test_matching_is_reproducible(self, palette):
        """Random restarts must not make the answer change between runs."""
        first = palette.match("#c08040", max_paints=2)
        second = palette.match("#c08040", max_paints=2)
        assert np.allclose(first.weights, second.weights)

    def test_ranked_matches_come_back_in_order(self, palette):
        recipes = palette.match_all("#8a9a5b", max_paints=2, limit=5)
        assert len(recipes) == 5
        assert recipes == sorted(recipes, key=lambda recipe: recipe.delta_e)
        assert recipes[0].delta_e <= recipes[-1].delta_e

    def test_the_best_ranked_match_is_what_match_returns(self, palette):
        best = palette.match("#8a9a5b", max_paints=2)
        ranked = palette.match_all("#8a9a5b", max_paints=2, limit=3)
        assert best.delta_e == pytest.approx(ranked[0].delta_e)

    def test_more_paints_never_matches_much_worse(self, palette):
        """A search over the whole palette can always fall back on any smaller recipe.

        The tolerance is there because the objective is not convex, so the larger search
        is not guaranteed to beat the smaller one -- but it should never be far behind.
        """
        few = palette.match("#8a9a5b", max_paints=3)
        many = palette.match("#8a9a5b")
        assert many.delta_e <= few.delta_e + 1.0

    def test_max_paints_below_one_is_refused(self, palette):
        with pytest.raises(ValueError, match="at least 1"):
            palette.match("#000000", max_paints=0)


class TestPaletteConstruction:
    def test_an_empty_palette_is_refused(self):
        with pytest.raises(ValueError, match="at least one paint"):
            Palette([])

    def test_repeated_names_are_refused(self):
        paint = Paint.from_srgb("Blue", "#0000ff")
        with pytest.raises(ValueError, match="repeated paint names"):
            Palette([paint, paint])

    def test_paints_can_be_looked_up_by_name_or_position(self, palette):
        assert palette["Viridian"] is palette[palette.names.index("Viridian")]

    def test_the_shipped_palette_admits_it_is_approximate(self, palette):
        assert palette.has_approximate_paints
        assert all(paint.is_approximate for paint in palette)


def test_a_target_can_be_given_in_cielab(palette):
    """CIELAB targets go through an explicit helper, since three numbers are read as sRGB."""
    from kubelka_munk import spectrum_from_lab, srgb_to_lab

    lab = srgb_to_lab(hex_to_srgb("#8a9a5b"))
    by_lab = palette.match(spectrum_from_lab(lab), max_paints=2)
    by_hex = palette.match("#8a9a5b", max_paints=2)
    assert by_lab.mixture.hex_colour == by_hex.mixture.hex_colour
