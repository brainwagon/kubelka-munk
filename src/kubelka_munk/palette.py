"""Palettes, the mixtures you can make from them, and the search for a recipe."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy.optimize import minimize

from .colorimetry import (
    delta_e_1976,
    delta_e_2000,
    lab_to_srgb,
    spectrum_to_lab,
    spectrum_to_srgb,
    srgb_to_hex,
)
from .kubelka_munk import apply_saunderson, opaque_reflectance
from .paint import Paint
from .spectrum import Spectrum
from .upsampling import reflectance_from_srgb

ColourDifference = Callable[[np.ndarray, np.ndarray], float]


@dataclass(frozen=True)
class Mixture:
    """Paints combined in stated proportions, and the colour that results.

    This is the forward direction: you say what to mix, this says what you get. Compare
    :class:`Recipe`, which is what a search *found*.
    """

    paints: tuple[Paint, ...]
    weights: np.ndarray
    spectrum: Spectrum

    @property
    def srgb(self) -> np.ndarray:
        """The mixed colour as sRGB in [0, 1]."""
        return spectrum_to_srgb(self.spectrum)

    @property
    def hex_colour(self) -> str:
        """The mixed colour as ``#rrggbb``."""
        return srgb_to_hex(self.srgb)

    @property
    def lab(self) -> np.ndarray:
        """The mixed colour in CIELAB."""
        return spectrum_to_lab(self.spectrum)

    def parts(self, minimum_weight: float = 1e-4) -> dict[str, float]:
        """The recipe as ``{paint name: proportion}``, dropping negligible amounts."""
        return {
            paint.name: float(weight)
            for paint, weight in zip(self.paints, self.weights)
            if weight >= minimum_weight
        }

    def __repr__(self) -> str:
        recipe = ", ".join(
            f"{name} {weight:.0%}" for name, weight in self.parts().items()
        )
        return f"<Mixture {self.hex_colour} = {recipe}>"


@dataclass(frozen=True)
class Recipe:
    """What a search found: proportions, the colour they give, and how close that is.

    ``delta_e`` is the CIEDE2000 difference from the target -- roughly, the number of
    just-noticeable differences between what you asked for and what you can mix. Below
    about 1 the two are indistinguishable side by side; below about 3 they are a good
    match; above 5 the target is out of reach of this palette.
    """

    mixture: Mixture
    target: Spectrum
    delta_e: float

    @property
    def weights(self) -> np.ndarray:
        return self.mixture.weights

    @property
    def paints(self) -> tuple[Paint, ...]:
        return self.mixture.paints

    def parts(self, minimum_weight: float = 1e-4) -> dict[str, float]:
        """The recipe as ``{paint name: proportion}``, dropping negligible amounts."""
        return self.mixture.parts(minimum_weight)

    @property
    def is_good_match(self) -> bool:
        """Whether the match is close enough to pass as the target in ordinary viewing."""
        return self.delta_e < 3.0

    def __repr__(self) -> str:
        recipe = ", ".join(
            f"{name} {weight:.0%}" for name, weight in self.parts().items()
        )
        return f"<Recipe {self.mixture.hex_colour} dE {self.delta_e:.2f} = {recipe}>"


class Palette:
    """The set of paints available to mix from.

    Mixing is Kubelka-Munk's additivity rule: the absorption and scattering of a mixture
    are the proportion-weighted sums of those of its ingredients. That single rule is
    what makes blue and yellow give green here, where blending the colours directly
    would give a dull grey.
    """

    def __init__(self, paints: Iterable[Paint]) -> None:
        self.paints = tuple(paints)
        if not self.paints:
            raise ValueError("a palette needs at least one paint")

        names = [paint.name for paint in self.paints]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise ValueError(f"palette has repeated paint names: {sorted(duplicates)}")

        self._absorptions = np.array([paint.absorption for paint in self.paints])
        self._scatterings = np.array([paint.scattering for paint in self.paints])
        self._surface_reflections = np.array(
            [paint.surface_reflection for paint in self.paints]
        )
        self._internal_reflections = np.array(
            [paint.internal_reflection for paint in self.paints]
        )

    def __len__(self) -> int:
        return len(self.paints)

    def __iter__(self):
        return iter(self.paints)

    def __getitem__(self, key: int | str) -> Paint:
        if isinstance(key, str):
            for paint in self.paints:
                if paint.name == key:
                    return paint
            raise KeyError(f"no paint named {key!r} in this palette")
        return self.paints[key]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(paint.name for paint in self.paints)

    @property
    def has_approximate_paints(self) -> bool:
        """Whether any paint's coefficients were guessed from a colour rather than measured."""
        return any(paint.is_approximate for paint in self.paints)

    def mix(self, weights: Sequence[float] | dict[str, float]) -> Mixture:
        """Mix these paints in the given proportions.

        Weights may be given positionally or as ``{paint name: parts}``, and need not
        sum to one -- ``{"Ultramarine": 2, "Titanium White": 1}`` is two parts blue to
        one part white, and is normalised for you.
        """
        weights = self._as_weight_vector(weights)
        return Mixture(
            paints=self.paints,
            weights=weights,
            spectrum=self._spectrum_of(weights),
        )

    def match(
        self,
        target: Spectrum | str | np.ndarray,
        max_paints: int | None = None,
        objective: ColourDifference = delta_e_2000,
        starts: int = 8,
        seed: int = 0,
    ) -> Recipe:
        """Find the proportions of this palette that come closest to a target colour.

        The target may be a :class:`Spectrum`, a hex string like ``"#6b8e23"``, or three
        sRGB values in [0, 1].

        Set ``max_paints`` to limit how many paints a recipe may use -- a painter mixing
        by hand wants three, not eleven. Every combination of that size is tried, so the
        answer is the true best subset, not a guess.

        The match is perceptual, not spectral: the result looks like the target under
        D65 daylight, and may drift under other lighting. Pass a spectral ``objective``
        if you need a match that holds under any light, or :func:`delta_e_1976` if you
        want a faster, cruder search.

        The recipe returned is the best one by the same measure reported in
        ``delta_e``, so the number you read is the number that was minimised.

        Always returns a Recipe. An unreachable target is not an error -- check
        ``delta_e`` to see how close the palette actually got.
        """
        return self.match_all(
            target,
            max_paints=max_paints,
            objective=objective,
            starts=starts,
            seed=seed,
            limit=1,
        )[0]

    def match_all(
        self,
        target: Spectrum | str | np.ndarray,
        max_paints: int | None = None,
        objective: ColourDifference = delta_e_2000,
        starts: int = 8,
        seed: int = 0,
        limit: int = 5,
    ) -> list[Recipe]:
        """Rank the best recipes for a target, closest first.

        With ``max_paints`` set this ranks the paint combinations against each other,
        which is what you want when the closest match uses colours you would rather not
        reach for. Without it there is only one recipe to rank, so the list holds one.
        """
        target_spectrum = as_spectrum(target)
        target_lab = spectrum_to_lab(target_spectrum)

        if max_paints is None or max_paints >= len(self.paints):
            subsets: list[tuple[int, ...]] = [tuple(range(len(self.paints)))]
        elif max_paints < 1:
            raise ValueError(f"max_paints must be at least 1, got {max_paints}")
        else:
            subsets = list(combinations(range(len(self.paints)), max_paints))

        recipes = [
            self._best_recipe_using(
                subset, target_spectrum, target_lab, objective, starts, seed
            )
            for subset in subsets
        ]
        recipes.sort(key=lambda recipe: recipe.delta_e)
        return recipes[: max(1, limit)]

    def _best_recipe_using(
        self,
        subset: tuple[int, ...],
        target_spectrum: Spectrum,
        target_lab: np.ndarray,
        objective: ColourDifference,
        starts: int,
        seed: int,
    ) -> Recipe:
        """Optimise the proportions of one fixed group of paints."""
        count = len(subset)

        def difference(subset_weights: np.ndarray) -> float:
            weights = np.zeros(len(self.paints))
            weights[list(subset)] = np.maximum(subset_weights, 0.0)
            total = weights.sum()
            if total <= 0.0:
                return float("inf")
            spectrum = self._spectrum_of(weights / total)
            return objective(spectrum_to_lab(spectrum), target_lab)

        if count == 1:
            best_weights = np.ones(1)
        else:
            best_weights, best_difference = None, float("inf")
            for start in self._starting_points(count, starts, seed):
                # The proportions must be non-negative and sum to one: they divide up a
                # fixed amount of paint. That is the simplex SLSQP is constrained to.
                result = minimize(
                    difference,
                    start,
                    method="SLSQP",
                    bounds=[(0.0, 1.0)] * count,
                    constraints=[
                        {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}
                    ],
                    options={"maxiter": 200, "ftol": 1e-10},
                )
                candidate = np.clip(result.x, 0.0, 1.0)
                if candidate.sum() <= 0:
                    continue
                candidate = candidate / candidate.sum()
                value = difference(candidate)
                if value < best_difference:
                    best_weights, best_difference = candidate, value
            if best_weights is None:
                best_weights = np.full(count, 1.0 / count)

        weights = np.zeros(len(self.paints))
        weights[list(subset)] = best_weights
        mixture = self.mix(weights)
        return Recipe(
            mixture=mixture,
            target=target_spectrum,
            delta_e=delta_e_2000(mixture.lab, target_lab),
        )

    @staticmethod
    def _starting_points(count: int, starts: int, seed: int) -> list[np.ndarray]:
        """Where to start the search from: the even mixture, then random ones.

        The objective is not convex, so a single start lands in a local minimum often
        enough to matter. The random starts are drawn from a fixed seed, so the same
        target always gives the same recipe.
        """
        points = [np.full(count, 1.0 / count)]
        if starts > 1:
            generator = np.random.default_rng(seed)
            # A flat Dirichlet draw is a uniformly random point on the simplex.
            points.extend(generator.dirichlet(np.ones(count), size=starts - 1))
        return points

    def _spectrum_of(self, weights: np.ndarray) -> Spectrum:
        """The reflectance of a mixture -- Kubelka-Munk's additivity rule, start to end.

        Absorption and scattering add in proportion; their ratio gives the reflectance of
        an opaque film; the Saunderson correction turns that into what you would see.
        """
        absorption = weights @ self._absorptions
        scattering = weights @ self._scatterings

        internal = opaque_reflectance(absorption / scattering)

        # The surface constants belong to the dried film, so a mixture takes the
        # weighted average of its ingredients'. When a palette shares one set of
        # constants -- the usual case -- this is just that set.
        measured = apply_saunderson(
            internal,
            float(weights @ self._surface_reflections),
            float(weights @ self._internal_reflections),
        )
        return Spectrum(measured)

    def _as_weight_vector(
        self, weights: Sequence[float] | dict[str, float]
    ) -> np.ndarray:
        """Normalise weights, given by position or by name, into proportions summing to one."""
        if isinstance(weights, dict):
            unknown = set(weights) - set(self.names)
            if unknown:
                raise KeyError(f"no paint named {sorted(unknown)} in this palette")
            vector = np.array(
                [float(weights.get(name, 0.0)) for name in self.names], dtype=float
            )
        else:
            vector = np.asarray(weights, dtype=float)
            if vector.shape != (len(self.paints),):
                raise ValueError(
                    f"expected one weight per paint ({len(self.paints)}), "
                    f"got {vector.shape[0] if vector.ndim else vector.shape}"
                )

        if np.any(vector < 0.0):
            raise ValueError("weights cannot be negative -- you cannot unmix a paint")
        total = vector.sum()
        if total <= 0.0:
            raise ValueError("weights are all zero -- there is nothing to mix")
        return vector / total

    def __repr__(self) -> str:
        return f"<Palette {len(self.paints)} paints: {', '.join(self.names)}>"


def as_spectrum(target: Spectrum | str | np.ndarray) -> Spectrum:
    """Accept a target colour however it was given and return a Spectrum.

    A hex string or sRGB triple has no one true spectrum, so a plausible one is
    reconstructed; see :mod:`kubelka_munk.upsampling`. This does not affect a
    perceptual match, which only compares the colours.

    Three numbers are always read as sRGB, never as CIELAB -- the two are not
    distinguishable by shape. Use :func:`spectrum_from_lab` for a CIELAB target.
    """
    if isinstance(target, Spectrum):
        return target
    if isinstance(target, str):
        return reflectance_from_srgb(target)

    values = np.asarray(target, dtype=float)
    if values.shape == (3,):
        return reflectance_from_srgb(values)
    return Spectrum(values)


def spectrum_from_lab(lab: np.ndarray) -> Spectrum:
    """A plausible reflectance curve for a colour given in CIELAB.

    CIELAB and sRGB triples are both three numbers, so a target given as a bare array is
    always read as sRGB. Call this first if your target is in CIELAB.
    """
    return reflectance_from_srgb(np.clip(lab_to_srgb(lab), 0.0, 1.0))
