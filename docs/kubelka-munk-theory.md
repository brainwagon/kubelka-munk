# Kubelka-Munk colour mixing: theory, sources, and what this library does with them

A review of the theory behind pigment mixing, and a map from that theory to the decisions
taken in this implementation. Sources are cited inline. Section 8 is the honest part: what
is approximated, what is invented, and what the literature simply does not provide.

The raw research notes this report is built from are kept at `notes/research-raw.md`, with
fuller citations and several passages quoted verbatim from the sources.

---

## 1. The problem with mixing colours

Averaging two colours is not mixing two paints. Blend the RGB values of ultramarine and
cadmium yellow and you get a muddy khaki; mix the paints and you get green. The averaging
is not a poor approximation of the mixing — it is a different operation, because paint
does not emit light, it removes light. Blue pigment absorbs long wavelengths, yellow
absorbs short ones, and green survives because neither absorbs the middle of the spectrum.
That reasoning is inescapably per-wavelength, which is why this library represents every
paint as a spectrum rather than as three numbers (ADR-0001).

## 2. The Kubelka-Munk model

Kubelka and Munk (1931) modelled a paint film as a turbid medium and tracked just two
fluxes through it: light travelling down into the film and light coming back up. Over an
infinitesimal depth, a fraction `K·dx` of a flux is absorbed and a fraction `S·dx` is
scattered into the opposite direction. `K` and `S` are the absorption and scattering
coefficients, both functions of wavelength, both per unit depth.

A recent treatment derives these two coupled equations from the three-dimensional
diffusion equation by lateral averaging, giving `K` and `S` a first-principles reading as
mean free paths ([Schirmacher & Ruocco 2023](https://arxiv.org/pdf/2303.04065)).

### 2.1 An opaque film

For a film thick enough that its backing cannot be seen, the model collapses to a single
relation between reflectance and the *ratio* `K/S` — the **remission function**:

```
K/S = (1 - R)² / 2R
```

and inverted,

```
R = 1 + K/S - sqrt((K/S)² + 2·K/S)
```

The other root of the quadratic gives `R > 1` and is discarded. Sources:
[Harrick Scientific's technical note](https://mmrc.caltech.edu/FTIR/Literature/Diff%20Refectance/Kubelka-Munk.pdf),
[Wikipedia](https://en.wikipedia.org/wiki/Kubelka%E2%80%93Munk_theory),
[Viggiano 2021](https://s3.cad.rit.edu/cadgallery_production/documents/2083/KM_Pathology.pdf).

These two functions are `absorption_over_scattering` and `opaque_reflectance` in
`kubelka_munk.py`, and they are the whole of the physics this library implements.

Note what the opaque case cannot express: only the *ratio* affects colour. A paint's
absolute `K` and `S` are invisible in its masstone. This is the fact that makes the
inverse problem in §5 hard, and it is the reason `Paint.from_srgb` has to guess.

### 2.2 A film you can see through

For a layer of finite thickness `X` over a backing of reflectance `Rg`, Kubelka's 1948
hyperbolic solution applies. With `a = (K + S)/S` and `b = sqrt(a² - 1)`:

```
R = [1 - Rg(a - b·coth(bSX))] / [a - Rg + b·coth(bSX)]
```

([Viggiano 2021](https://s3.cad.rit.edu/cadgallery_production/documents/2083/KM_Pathology.pdf), Eq. 2.)
As `X → ∞` this reduces to §2.1. This is what glazing, scumbling and watercolour need, and
this library does not implement it (ADR-0007). Storing `K` and `S` separately rather than
only their ratio means adding it later is an extension rather than a rewrite.

### 2.3 The mixing rule

Everything the library does rests on one line. When paints are mixed in proportions `cᵢ`,
their coefficients add:

```
K_mix(λ) = Σᵢ cᵢ Kᵢ(λ)
S_mix(λ) = Σᵢ cᵢ Sᵢ(λ)
```

([Centore 2020](https://onlinelibrary.wiley.com/doi/abs/10.1111/cote.12497).) Compute the
mixture's `K` and `S`, take the ratio, apply the remission function, and you have the
colour. That is `Palette._spectrum_of` in its entirety.

A common implementation mistake, flagged explicitly in
[Viggiano 2021](https://s3.cad.rit.edu/cadgallery_production/documents/2083/KM_Pathology.pdf),
is to interpolate `K/S` ratios directly instead of carrying `K` and `S` separately. `K/S`
is unbounded above and mixing it linearly is simply not the model.

## 3. One constant or two

**Two-constant** theory carries `K` and `S` independently per paint. **Single-constant**
theory keeps only `K/S`, halving the measurements needed, and is common in art
conservation where measuring every pigment twice is impractical.

The simplification has a documented failure mode, and it is precisely the case painters
care about: it "can lead to errors in pigment selection for dark colors and colors not
containing a white pigment"
([Zhang et al. 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9283763/)). White works by
scattering enormously while absorbing almost nothing — the very thing single-constant
theory throws away. Since mixing with white is most of what a painter does, this library
is two-constant throughout (ADR-0005).

Single-constant theory also has no agreed normalisation. Because `K/S` is not linear in
concentration in the way `K` and `S` separately are, practitioners need a tinting-strength
convention to make different pigments' numbers comparable, and the research turned up no
canonical specification for one — it appears to be handled ad hoc.

## 4. The surface: Saunderson's correction

Kubelka-Munk assumes light enters the film freely. A real surface reflects some light
straight back off the top without it ever meeting a pigment, and reflects some back down
again from the inside. Saunderson's correction relates internal reflectance `R` to
measured reflectance `Rm` through two constants:

```
Rm = k₁ + (1 - k₁)(1 - k₂)·R / (1 - k₂·R)
```

([Saunderson 1942](https://en.wikipedia.org/wiki/Kubelka%E2%80%93Munk_theory); formula
confirmed verbatim against the
[MunsellAndKubelkaMunkToolbox source](https://github.com/colour-science/MunsellAndKubelkaMunkToolbox/blob/master/KubelkaMunk/SaundersonCorrection.m).)
Ignoring `k₂` makes dark mixtures come out too light. The library applies the correction
at exactly two boundaries — measured to internal on the way in, internal to measured on
the way out (ADR-0006).

### 4.1 Why k₁ defaults to zero here

Published values vary: `k₁` from 0.04 to 0.08, `k₂` from 0.4 to 0.6. This library defaults
`k₂` to 0.6 and `k₁` to **zero**, which is a deliberate departure worth explaining.

`k₁` is the light glinting off the surface. That is exactly what a spectrophotometer
discards when measuring with the specular component excluded — the usual geometry for
paint — and it is likewise absent from a colour named as an sRGB value, which says how a
surface looks and not what is reflecting off it. Setting `k₁ = 0.04` alongside such data
counts the surface twice, and imposes a floor: with 4% of the light returning off the top,
nothing can appear darker than 4% reflectance.

This was not a theoretical worry. Building the shipped palette with `k₁ = 0.04` turned
ivory black `#23201e` into `#383838` and phthalo blue `#0d3b66` into `#323d66` — every
dark paint clamped to a washed-out grey, and blue-plus-yellow stopped producing green.
With `k₁ = 0`, all twelve masstones round-trip exactly. Pass `k₁ = 0.04` when your
measurements genuinely include the specular component.

## 5. Getting coefficients for real paints

### 5.1 The honest way: masstone and tint

Absorption and scattering cannot be separated from one measurement of an opaque film,
since only their ratio shows. Adding a known white in a known proportion breaks the tie:
the white contributes scattering you already know, so the degree to which the tint lightens
reveals how much the paint itself scatters. A masstone plus one tint suffices in principle
to recover both `K` and `S` at every wavelength
([Zhang et al. 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9283763/)).

This is `Paint.from_measurements`. Writing `c` for the tint fraction, `r` for the tint's
`K/S` and `m` for the masstone's, the mixing rule gives one linear equation per wavelength:

```
S = (1-c)(r·S_white - K_white) / (c·(m - r))
K = m·S
```

Scattering is clamped at zero, since measurement noise can otherwise drive it negative,
which is unphysical — `K` and `S` cannot be negative by the first law of thermodynamics
([Viggiano 2021](https://s3.cad.rit.edu/cadgallery_production/documents/2083/KM_Pathology.pdf)).

The test suite verifies this by mixing a paint of known `K` and `S` with a known white,
then calibrating from the result and checking the original coefficients come back.

### 5.2 The available way: one hex code

Most users have no spectrophotometer. `Paint.from_srgb` takes a single colour and does two
things that are not measurements, and it is important to be clear about which is which.

**First, a spectrum from a colour.** Infinitely many reflectance curves produce the same
sRGB, so this is ill-posed. The literature offers
[Smits 1999](https://github.com/colour-science/smits1999),
[Meng et al. 2015](https://jo.dreggn.org/home/2015_spectrum.pdf),
[Jakob & Hanika 2019](https://jo.dreggn.org/home/2019_sigmoid.pdf),
[Otsu et al. 2018](https://cs.uwaterloo.ca/~thachisu/rgb2spec.pdf) and
[Mallett & Yuksel 2019](http://www.cemyuksel.com/research/papers/spectral_primary_decomposition.pdf).
This library uses Scott Burns's least-slope-squared family, specifically the
hyperbolic-tangent variant (LHTSS), which selects the *smoothest* curve reproducing the
target colour exactly, with the substitution `R = (1 + tanh z)/2` confining reflectance to
(0, 1) by construction rather than by clamping
([Burns, arXiv:1710.05732](https://arxiv.org/pdf/1710.05732);
[Burns 2020](http://scottburns.us/wp-content/uploads/2025/03/Numerical-Methods-for-Smoothest-Reflectance-Reconstruction-Burns-2020.pdf)).
It ships no data table and its derivation fits in a docstring (ADR-0008). Smoothness is a
real property of pigment reflectances, so "smoothest curve with this colour" is a
defensible principle rather than an arbitrary tie-break.

*(A note on attribution: the paper titled "Spectral Primary Decomposition" is by Mallett &
Yuksel, not Burns. The two bodies of work are unrelated and are easy to conflate.)*

**Second, a split of `K/S` into `K` and `S`.** This one is a fiction. The library assumes
scattering is flat across wavelength and equal to a per-paint `tinting_strength`, then
takes `K = (K/S)·S`. Nothing about a colour justifies this; it is one clearly-labelled
dial standing in for a measurement. It is worth setting, because it is what makes an
opaque cadmium and a transparent quinacridone behave differently against white — and
because leaving it at its default makes every paint on a palette tint alike, which no real
palette does.

Since the masstone fixes `K/S`, raising `tinting_strength` raises `K` and `S` together and
leaves the paint's own colour untouched while increasing its power in a mixture. Paints
built this way are flagged `is_approximate`.

## 6. From spectrum to colour

Reflectance becomes a colour by integrating against an illuminant and the standard
observer:

```
X = k · Σ E(λ)·R(λ)·x̄(λ)      (likewise Y and Z)
k = 1 / Σ E(λ)·ȳ(λ)
```

with `k` chosen so a perfect reflecting diffuser has `Y = 1`. The library uses the CIE 1931
2° observer and illuminant D65, tabulated at 10 nm over 380–730 nm from the
[CVRL](http://www.cvrl.org) republication of the CIE data, embedded in `cie_data.py`
rather than pulled from a dependency. A test asserts that a perfect white reflector lands
on the D65 white point; at 10 nm decimation it gives (0.9501, 1.0, 1.0882) against the
nominal (0.9505, 1.0, 1.0890).

XYZ becomes sRGB through the IEC 61966-2-1 matrix and transfer function, and CIELAB
through the standard 1976 definition
([sRGB](https://en.wikipedia.org/wiki/SRGB)).

### 6.1 Colour difference

CIE76 is Euclidean distance in CIELAB. CIEDE2000 adds a chroma-dependent rescaling of
`a*`, weighting functions for lightness, chroma and hue, and a rotation term correcting
CIELAB's distortion in the blue region; a value near 1 is about one just-noticeable
difference.

CIEDE2000 is intricate enough that reconstructing it from prose is unwise. This
implementation follows Sharma, Wu and Dalal (2005) and is pinned against that paper's
34-pair test set, which ships in `tests/ciede2000_test_data.txt` and is fetched from
[the authors' site](https://hajim.rochester.edu/ece/sites/gsharma/ciede2000/). Those pairs
are chosen to exercise the hue discontinuity at 360°, the blue rotation term and
near-neutral colours; all 34 agree to within the test file's four-decimal rounding.

## 7. The inverse problem: finding a recipe

Predicting a mixture's colour is a formula. Going backwards — finding proportions that hit
a target — is an optimisation, and the paint industry has worked on it since
[Allen's algorithm](https://onlinelibrary.wiley.com/doi/abs/10.1111/cote.12497) in the
1960s. The modern framing is a constrained optimisation over the simplex.

This library minimises colour difference over weights that are non-negative and sum to one
— a partition of a fixed volume of paint (ADR-0004) — using SLSQP from multiple starting
points. The objective is not convex, so a single start lands in a local minimum often
enough to matter; the restarts are drawn from a fixed seed so the same target always gives
the same recipe (ADR-0009). A `max_paints` limit is honoured by enumerating every subset
of that size and solving each, which is exact and, for a palette of a dozen paints, cheap.

### 7.1 Spectral versus perceptual matching, and a correction

There are two things "match" can mean. A **spectral** match reproduces the target's
reflectance curve and therefore holds under any illuminant. A **tristimulus** match only
has to look the same under one, and is usually all that is achievable — the resulting pair
is a metamer, agreeing under D65 and potentially parting company under tungsten.

This library matches perceptually, since a painter wants "looks right", and exposes the
objective so a spectral criterion can be substituted.

The original design used CIE76 as the objective — cheap and smooth, good for a gradient
solver — while reporting CIEDE2000. Implementation showed this to be wrong. The metrics do
not merely differ in scale, they disagree about which recipe is better: matching `#8a9a5b`
from titanium white and yellow ochre, CIE76 is minimised at 65% white and CIEDE2000 at 0%
white. The library was returning a recipe that was not the best one by the number printed
beside it, and rankings were not monotonic. Driving the search with CIEDE2000 directly
costs about 30% more time and improves the matches measurably — summed CIEDE2000 over a
sample of eight targets fell from 27.4 to 22.3 — and its discontinuities did not trouble
SLSQP in practice. ADR-0002 records the reversal.

An unreachable target is data, not an error: `match` always returns a recipe and its
`delta_e`. Pure display green is simply not available from pigments, and the library says
so rather than silently returning a yellow-green as though it had succeeded.

## 8. What is approximate, and what is missing

**No open dataset of Kubelka-Munk coefficients for artists' paints was found.** This is the
central practical gap. Reflectance-spectrum datasets do exist — the
[Pigments Checker](https://chsopensource.org/chsos-application-note-4/) and an
[artist acrylic paint dataset](https://library.imaging.org/admin/apis/public/api/ist/website/downloadArticle/archiving/19/1/10)
— and `K`/`S` could in principle be derived from them where masstone and tint pairs are
present, but the coefficients used by commercial colour-matching systems are not published.
[Mixbox](https://github.com/scrtwpns/mixbox) (Sochorová & Jamriška 2021) exists partly
because of this: it bakes measurements of real pigments into a precomputed latent space
with an RGB-in, RGB-out interface, rather than exposing spectral data. It is worth knowing
about as the practical alternative to this library, and it is CC BY-NC licensed.

Consequently `APPROXIMATE_ARTIST_PALETTE` is a plausible toy, not data. Each entry is a hex
code eyeballed from a colour chart plus a tinting strength set by judgement.

**Other limits**, each recorded in an ADR: opaque films only, so no glazing; matches are
metameric; `from_srgb` invents scattering; and the sRGB round-trip holds targets a hair
inside the gamut because the `tanh` parametrisation reaches 0 and 1 only in the limit.

**Numerical care.** Reflectance is clamped into (0, 1): `K/S` diverges as `R → 0`, and
values at or above 1 are unphysical for a non-fluorescent surface. Scattering recovered
from measurements is clamped at zero. In the reflectance-recovery solve, the Newton system
becomes singular where `tanh` saturates, so it is solved by least squares rather than a
direct factorisation — which is what lets pure white and pure black work at all.

**Kubelka-Munk is itself an approximation.** Two fluxes, a uniform medium, no account of
particle size or binder. It is a good working model of how paint mixes, not a physical
description of it.

---

## Sources

Core theory:

- [Viggiano, "Numerical pathology in selected Kubelka-Munk formulas" (2021)](https://s3.cad.rit.edu/cadgallery_production/documents/2083/KM_Pathology.pdf) — modern notation, the hyperbolic forms, and the failure modes
- [Schirmacher & Ruocco, "Diffusion of light in turbid media and Kubelka-Munk theory" (2023)](https://arxiv.org/pdf/2303.04065)
- [Harrick Scientific, "What is Kubelka-Munk?"](https://mmrc.caltech.edu/FTIR/Literature/Diff%20Refectance/Kubelka-Munk.pdf)
- [Wikipedia: Kubelka-Munk theory](https://en.wikipedia.org/wiki/Kubelka%E2%80%93Munk_theory)

One constant versus two, and calibration:

- [Zhang et al., "Optimal Learning Samples for Two-Constant Kubelka-Munk Theory" (2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9283763/)
- [Rich & Martínez, "On the Kubelka-Munk Single-Constant/Two-Constant Theories"](https://scispace.com/pdf/on-the-kubelka-munk-single-constant-two-constant-theories-4ojksvvlv2.pdf)
- [Centore, "Enforcing Kubelka-Munk constraints for opaque paints" (2020)](https://onlinelibrary.wiley.com/doi/abs/10.1111/cote.12497)

Surface correction:

- [MunsellAndKubelkaMunkToolbox, `SaundersonCorrection.m`](https://github.com/colour-science/MunsellAndKubelkaMunkToolbox/blob/master/KubelkaMunk/SaundersonCorrection.m)

Spectral upsampling:

- [Burns, "Generating Reflectance Curves from sRGB Triplets" (arXiv:1710.05732)](https://arxiv.org/pdf/1710.05732)
- [Burns, "Numerical Methods for Smoothest Reflectance Reconstruction" (2020)](http://scottburns.us/wp-content/uploads/2025/03/Numerical-Methods-for-Smoothest-Reflectance-Reconstruction-Burns-2020.pdf)
- [Jakob & Hanika (2019)](https://jo.dreggn.org/home/2019_sigmoid.pdf), [Meng et al. (2015)](https://jo.dreggn.org/home/2015_spectrum.pdf), [Otsu et al. (2018)](https://cs.uwaterloo.ca/~thachisu/rgb2spec.pdf), [Mallett & Yuksel (2019)](http://www.cemyuksel.com/research/papers/spectral_primary_decomposition.pdf)

Colorimetry:

- [CVRL colour matching functions](http://www.cvrl.org/database/data/cmfs/ciexyz31.csv) and [illuminant D65](http://www.cvrl.org/database/data/cie/Illuminantd65.csv)
- [sRGB matrices and transfer function](https://en.wikipedia.org/wiki/SRGB)
- [Sharma, Wu & Dalal, CIEDE2000 implementation notes and test data](https://hajim.rochester.edu/ece/sites/gsharma/ciede2000/)

Prior art and data:

- [Mixbox: Practical Pigment Mixing for Digital Painting (2021)](https://github.com/scrtwpns/mixbox) — [paper](https://dcgi.fel.cvut.cz/wp-content/wpallimport-dist/publications/pdf/publications-2021-sochorova-tog-pigments-paper.pdf)
- [Pigments Checker reflectance database](https://chsopensource.org/chsos-application-note-4/)
- [Artist Acrylic Paint Spectral and Colorimetric Dataset](https://library.imaging.org/admin/apis/public/api/ist/website/downloadArticle/archiving/19/1/10)
