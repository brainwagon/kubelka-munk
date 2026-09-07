# Kubelka-Munk Color Mixing: Research Notes

Raw research notes for a Python library that (a) mixes artists' paints spectrally using Kubelka-Munk (K-M) theory and (b) solves the inverse problem of finding palette proportions that best match a target color. Compiled from web search and primary/secondary sources; every substantive claim is cited inline. Where sources disagree or data could not be found, this is flagged explicitly.

---

## 1. Core Kubelka-Munk equations

### 1.1 Origin and setup

Kubelka and Munk (1931) modeled light transport in a turbid (scattering + absorbing) layer with two coupled first-order linear differential equations for a downward flux and an upward (reflected) flux traveling through a plane-parallel layer, each attenuated by an absorption coefficient `K` and a (back-)scattering coefficient `S` per unit thickness ("the proportion of light absorbed/scattered in a layer of infinitesimal thickness dx is K·dx / S·dx"). This formulation, and the notation, is summarized precisely (with historical citations to Kubelka's 1948 paper and Gurevič 1930) in J.A. Stephen Viggiano, *"Numerical pathology in selected Kubelka-Munk formulas, and strategies for mitigation"* (2021) — https://s3.cad.rit.edu/cadgallery_production/documents/2083/KM_Pathology.pdf

A recent theoretical treatment shows the K-M equations are mathematically equivalent to a one-dimensional radiative-diffusion equation obtained by laterally averaging the 3-D diffusion equation, giving the K and S parameters a first-principles physical interpretation (mean free paths for absorption/scattering/transport): Schirmacher & Ruocco, *"Diffusion of light in turbid media and Kubelka-Munk theory"* (arXiv, 2023) — https://arxiv.org/pdf/2303.04065

### 1.2 K/S from reflectance (the remission function)

For an **opaque** (infinitely/semi-infinitely thick) layer, the K-M model gives a closed-form relation between the diffuse reflectance `R∞` and the ratio `K/S`, known as the **remission function** or **K-M transform**:

```
K/S = (1 - R∞)^2 / (2 R∞)
```

This is confirmed by multiple sources verbatim, e.g. Harrick Scientific's "What is Kubelka-Munk?" technical note (a widely cited practitioner explainer used in FTIR diffuse-reflectance spectroscopy) — https://mmrc.caltech.edu/FTIR/Literature/Diff%20Refectance/Kubelka-Munk.pdf (fetched and OCR'd directly: "the K-M transform" `k/s = (1-R∞)²/2R∞`), and by the ScienceDirect Kubelka-Munk Theory topic overview — https://www.sciencedirect.com/topics/engineering/kubelka-munk-theory

This function is monotonic on `R∞ ∈ (0,1]` and is the standard way spectrophotometric reflectance is converted to an (approximately) concentration-proportional quantity, analogous to absorbance in transmission spectroscopy (Harrick note, same URL, notes K/S "is approximately proportional to the concentration").

### 1.3 Reflectance of an opaque layer from K/S (the inverse direction)

Inverting the quadratic above gives the reflectance of an infinitely thick layer from the K/S ratio:

```
R∞ = 1 + (K/S) - sqrt[ (K/S)^2 + 2(K/S) ]
```

This exact form appears in the Wikipedia Kubelka–Munk theory article (fetched directly) — https://en.wikipedia.org/wiki/Kubelka%E2%80%93Munk_theory — written there in terms of "absorption fraction `a₀`" and "remission fraction `r₀`" as `R∞ = 1 + a₀/r₀ - sqrt[(a₀/r₀)² + 2(a₀/r₀)]`, which is the same equation with `a₀/r₀ ≡ K/S`. It is also given in this form in web search results tied to patent literature and Viggiano's notation table (`R∞ = 1 + K/S − sqrt[(K/S)² + 2K/S]`), corroborating the sign/branch choice (the other root of the quadratic is unphysical, giving `R∞ > 1`).

### 1.4 Finite-thickness ("translucent") layer over a substrate

For a layer of finite thickness `X` sitting on a backing/substrate of reflectance `Rg`, Kubelka's 1948 paper gives (and Viggiano's 2021 paper restates in modern notation, extracted directly from the PDF):

Define:
```
a = (K + S) / S            (so a >= 1)
b = sqrt(a^2 - 1)
```

Then Kubelka's **hyperbolic solution** for reflectance is:

```
R = [ 1 - Rg (a - b·coth(bSX)) ] / [ a - Rg + b·coth(bSX) ]
```

— Viggiano (2021), Eq. (2), https://s3.cad.rit.edu/cadgallery_production/documents/2083/KM_Pathology.pdf (text extracted verbatim via PDF text layer). The corresponding transmittance is:

```
T = b / ( a·sinh(bSX) + b·cosh(bSX) )
```

(Viggiano, Eq. (5), same source), and the reflectance over a **perfectly black backing** (`Rg = 0`), sometimes called `R0`, is:

```
R0 = 1 / ( a + b·coth(bSX) )
```

(Viggiano, Eq. (7), same source; an algebraically equivalent non-singular form multiplies through by `sinh(bSX)`: `R0 = sinh(bSX) / [a·sinh(bSX) + b·cosh(bSX)]`, Viggiano Eq. 7a).

An equivalent (older, exponential rather than hyperbolic) form for general `R` over substrate `Rg`, from Kubelka & Munk's original 1931 paper as restated by Viggiano, Eq. (1):

```
R = [ (Rg - R∞)/R∞ - R∞(Rg - 1/R∞)·exp(SX(1/R∞ - R∞)) ]
    -----------------------------------------------------
    [ Rg - R∞ - (Rg - 1/R∞)·exp(SX(1/R∞ - R∞)) ]
```

Viggiano's paper explicitly discusses this form as being numerically worse-behaved (four subtractions, `1/R∞` blows up as `S → 0`) than the hyperbolic form — see §9 below on numerical pitfalls.

As `X → ∞`, all of these reduce to the opaque-layer formula in §1.3. Note that per Viggiano's units table, `K`, `S` (and `L`, `M`, `P` derived quantities) have units of inverse length (reciprocal of whatever unit `X` is measured in — customarily micrometers, though mass per unit area, e.g. g/m², is sometimes used as a thickness surrogate when density is known); `R`, `T`, `a`, `b` are dimensionless (same source).

---

## 2. Single-constant vs. two-constant K-M theory

### 2.1 Two-constant theory

**Two-constant K-M** treats `K` (absorption) and `S` (scattering) as two independent, wavelength-dependent coefficients per colorant, which must both be determined from measurement. It is considered the more accurate/general model. A 2022 study states: "The two-constant Kubelka-Munk theory has the highest prediction accuracy and moderate learning sample size requirement among all the color prediction models," and describes the standard **two-sample calibration**: a masstone (100% colorant) sample plus one tint (colorant + white in known ratio, e.g. 40:60) suffice in principle to solve for both `K` and `S` of a colorant at each wavelength — Zhang et al., *"Optimal Learning Samples for Two-Constant Kubelka-Munk Theory to Match the Color of Pre-colored Fiber Blends"*, Frontiers/PMC (2022) — https://pmc.ncbi.nlm.nih.gov/articles/PMC9283763/ , also https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.945454/full . A classic reference: "Kubelka and Munk absorption and scattering coefficients for a nonwhite pigment dispersion can be obtained from reflectance measurements on opaque samples of a reference white base and two tinter/white mixtures" (search-result summary of standard colorant-formulation literature, corroborated by Allen's papers, §6 below).

The historically important theoretical paper distinguishing the two regimes is P. Kubelka, *"New Contributions to the Optics of Intensely Light-Scattering Materials, Part II: Non-Homogeneous Layers"* and the debate is treated at length in Rich & Martínez, *"On the Kubelka-Munk Single-Constant/Two-Constant Theories"*, Color Research & Application — https://www.researchgate.net/publication/216567998_On_the_Kubelka-Munk_Single-ConstantTwo-Constant_Theories (PDF mirror: https://scispace.com/pdf/on-the-kubelka-munk-single-constant-two-constant-theories-4ojksvvlv2.pdf).

### 2.2 Single-constant theory

**Single-constant K-M** assumes `S` is the same (or irrelevant/cancels) for all colorants and only the ratio `K/S` per colorant (or just `K`, with a shared/reference `S`) is tracked. This drastically reduces the number of measurements needed. It is described as commonly used in art conservation: "Simplified methods have been developed for art conservation where a single tint of each pigment mixed with white is required to define a pigment's optical properties based on the single-constant form of Kubelka-Munk turbid media theory" — search-result summary tied to the fiber-blend literature above (PMC9283763). The tradeoff is explicit: "This simplification can lead to errors in pigment selection for dark colors and colors not containing a white pigment" (same source). A dedicated review is R. Balasubramanian / or similarly-titled: *"Single-constant simplification of Kubelka-Munk turbid-media theory for paint systems — A review"* — https://www.researchgate.net/publication/227684964_Single-constant_simplification_of_Kubelka-Munk_turbid-media_theory_for_paint_systems-A_review (abstract-level only; full text not fetched — noted as a gap).

### 2.3 Mixture rule: linear combination by concentration

Under K-M theory, when several colorants are physically mixed, the **K and S of the mixture are the concentration-weighted linear sums** of the individual colorants' K and S values at each wavelength:

```
K_mix(λ) = Σ_i c_i · K_i(λ)
S_mix(λ) = Σ_i c_i · S_i(λ)
```

This additivity (with `c_i` being the volume or weight fraction, or "concentration," of colorant `i`) is the foundational assumption that makes K-M tractable for recipe prediction — stated explicitly, e.g., in Centore's 2020 abstract: "The Kubelka–Munk model relates the colours of paint mixtures to the absorption and scattering coefficients (K and S) of the constituent paints, and to their concentrations (C) in the mixtures" — Centore, *"Enforcing Kubelka–Munk constraints for opaque paints"*, Coloration Technology (2020) — https://onlinelibrary.wiley.com/doi/abs/10.1111/cote.12497 . It is also implicit throughout the two-constant calibration literature in §2.1 (each colorant's `K_i, S_i` are per-unit-concentration coefficients that are summed, weighted by fraction present, to predict a mixture's `K/S` and hence its `R∞`).

### 2.4 "Unit concentration" and the normalization problem for single-constant theory

"Unit concentration" refers to normalizing each colorant's measured `K` (or `K/S`) to a reference amount (e.g., 1 g pigment per unit area of substrate, or 100% strength), so that different colorants' coefficients can be linearly combined at arbitrary mixing ratios. In full two-constant theory, both `K_i` and `S_i` are measured per unit concentration and this is unambiguous. In **single-constant** theory, only a `K/S`-like scalar per colorant is tracked, but K/S is *not* itself linear in concentration in the same simple additive way once you fold `S` implicitly into a shared/reference scattering base — practitioners therefore need a **tinting-strength convention**: an agreed reference recipe (e.g., a fixed mass fraction of colorant in a fixed white base) against which every pigment's relative strength ("unit concentration") is calibrated, so that mixing ratios of different pigments' single-constant "K" values remain comparable. This convention issue is exactly why single-constant methods are reported to "lead to errors in pigment selection for dark colors and colors not containing a white pigment" (PMC9283763, above) — because outside the calibration regime (near-black mixtures, or mixtures with no white), the implicit shared-`S` assumption breaks down and the tinting-strength normalization no longer holds. *(Note: I could not find one single canonical "tinting strength convention" specification — this appears to be handled ad hoc / per-practitioner in the conservation and art-material literature; flagged as a genuine gap.)*

---

## 3. The Saunderson correction

### 3.1 Why it's needed

The K-M model as derived assumes **no reflection at the top (air/film) surface** — light either enters the medium completely or is fully diffusely reflected out of it according to K and S alone. Physically this is false: at any refractive-index discontinuity (air to paint film, roughly n≈1.5), some light is specularly reflected at the surface (Fresnel reflection) without ever "seeing" the pigment, and some light traveling inside the film that hits the interior of the interface is reflected back into the film rather than escaping (total-internal-reflection-like effect at oblique angles integrated over a diffuse field). This is stated in search-result summaries of the ScienceDirect Kubelka-Munk topic page and corroborated by the correction's namesake paper: J.L. Saunderson, *"Calculation of the Color of Pigmented Plastics"*, JOSA (1942) — original correction; see also the peer-reviewed reassessment R. Cortés / or similarly, *"An assessment of Saunderson corrections to the diffuse reflectance of paint films"* — https://www.researchgate.net/publication/231037127_An_assessment_of_Saunderson_corrections_to_the_diffuse_reflectance_of_paint_films

### 3.2 The correction formula

The correction relates the **measured** (instrument-observed) reflectance `Rm` to the **internal** K-M reflectance `R∞` (or `R`) via two Fresnel-type constants `K1` and `K2`:

```
Rm = K1 + (1 - K1)(1 - K2) R∞ / (1 - K2·R∞)
```

Confirmed directly from source code in the `colour-science` project's MunsellAndKubelkaMunkToolbox, `SaundersonCorrection.m` (fetched verbatim) — https://github.com/colour-science/MunsellAndKubelkaMunkToolbox/blob/master/KubelkaMunk/SaundersonCorrection.m — implemented literally as `R_m = K1 + ((1-K1)*(1-K2)*R_Inf)./(1-(K2.*R_Inf))`. Per that source's in-code documentation:
- **K1**: "the fraction ... of the light impinging (from OUTside the film) on a paint or ink, that is reflected back directly from the paint surface, without first entering the paint film" (external/specular Fresnel reflectance).
- **K2**: "the fraction ... of the light impinging (from INside the film) on the interface between the film and the surrounding air ... that is reflected back into the film" (internal diffuse reflectance at the interface).

### 3.3 Typical values

- **K1** is computable from Fresnel's equations given the refractive indices of air and the film/binder (no single fixed constant — depends on wavelength and angle-of-incidence geometry of the instrument), per the toolbox docs above.
- **K2** is harder to measure directly; the toolbox docs note "values as low as 0.4 might be used," with a commonly cited theoretical value of about **0.6** for a perfectly diffuse internal light field, and practical usage in the range **0.4–0.6** (same GitHub source).
- A separate web-search summary of general Saunderson-correction practice cites commonly used rule-of-thumb values **k1 ≈ 0.08** and **k2 ≈ 0.5** — search-result synthesis referencing multiple patent/paper sources (e.g. https://www.sciencedirect.com/topics/engineering/kubelka-munk-theory, https://onlinelibrary.wiley.com/doi/abs/10.1111/cote.12497 lineage) — flagged here as a source disagreement: the exact numeric convention (0.04 vs 0.08 for k1, common assumption for a glossy vs matte finish) varies significantly across the literature and by instrument geometry (specular-included vs specular-excluded measurement), so a real implementation should treat these as tunable/measured, not hard-coded physical constants.

### 3.4 Why it matters before/after inverting K-M

Since the K-M remission function `K/S = (1-R)²/(2R)` (§1.2) operates on the **internal** reflectance `R∞`, not the instrument-measured `Rm`, the Saunderson correction must be **inverted** (solve the above for `R∞` given a measured `Rm`) *before* computing `K/S` from a real spectrophotometer reading, and it must be **applied forward** (as shown above) *after* predicting an internal `R∞` from mixed `K/S` values, in order to predict what an instrument (or the eye, roughly) will actually observe. Skipping this step is a common source of poor K-M color-matching accuracy in practice, especially for glossy/varnished paints where the surface term is large.

---

## 4. Determining pigment K and S in practice

### 4.1 The masstone + tint procedure (two-constant calibration)

The standard practical calibration for two-constant K-M per colorant:
1. Prepare an **opaque masstone** sample: the colorant alone (or effectively "hiding," i.e., thick enough that the substrate underneath doesn't show through), measure its reflectance spectrum `R_mt(λ)`, and compute `(K/S)_mt = (1-R_mt)²/(2R_mt)` at each wavelength.
2. Prepare one or more **tints**: known-ratio mixtures of the colorant with a well-characterized reference white (whose own `K_w(λ), S_w(λ)` are separately known/measured), also opaque, and measure their reflectance.
3. Because `K_mix = c·K_pigment + (1-c)·K_white` and `S_mix = c·S_pigment + (1-c)·S_white` (§2.3), and each tint gives one `(K/S)` equation via its measured `R∞`, two independent samples (masstone + one tint, or two tints of different ratio) give two equations that can be solved simultaneously for the pigment's own `K(λ)` and `S(λ)` at each wavelength, once the white's `K_w, S_w` are known from its own calibration against a perfect diffuser or known-`Rg` substrate.

This is exactly the procedure referenced above: "reflectance measurements on opaque samples of a reference white base and two tinter/white mixtures" and "[a] masstone obtained by 100% pre-colored fiber and a tint mixed by 40% pre-colored fiber and 60% white fiber[] are enough to determine the absorption and scattering coefficients" — Zhang et al. (2022), https://pmc.ncbi.nlm.nih.gov/articles/PMC9283763/ . Allen's foundational computer-color-matching papers formalize the underlying linear-algebra approach (see §6).

### 4.2 What a practitioner can do with only sRGB swatches

If only sRGB (three-channel, non-spectral) swatch data is available — as is realistic for most digital-only artist-paint references — a physically rigorous two-constant K/S determination is **not possible**: K-M requires a full reflectance spectrum (or, at minimum, several spectral bands) per sample, because `K(λ)` and `S(λ)` are wavelength-dependent and 3 RGB numbers under-constrain a spectrum. Practically available workarounds, all approximate:
- **Spectral upsampling** an sRGB swatch to a plausible reflectance curve (see §7) and then treating that reconstructed curve as if it were measured, to get an approximate `K/S(λ)` via the standard transform. This inherits all the ambiguity/metamerism of the upsampling method used and cannot recover the *true* pigment spectrum, only *a* plausible metamer.
- **Single-constant approximation**: treat sRGB-derived `K/S` (from an upsampled or even a coarse 3-band "spectrum") as the colorant's only free parameter, sharing one fixed `S(λ)` (often a flat/constant `S=1`) across all colorants — this is the practical basis of the "art conservation" single-tint approach in §2.2, extended to sRGB-only.
- Accept and document that predictions in this regime are for *plausible-looking* mixing behavior (hue shifts, darkening, etc.) rather than laboratory-grade spectral color matching — this is essentially the stance taken by Mixbox (§8) and by graphics-oriented "subtractive mixing" papers like Burns's — S.A. Burns, *"Subtractive Color Mixture Computation"*, arXiv:1710.06364 — https://arxiv.org/pdf/1710.06364 — which explicitly frames full K-M with "extensive spectrophotometric measurements" as accurate but impractical for graphics use, trading precision for a measurement-free approximate method.

---

## 5. Spectral pipeline (wavelengths → XYZ → sRGB → CIELAB → ΔE)

### 5.1 Wavelength sampling

Two conventions dominate:
- **380–730 nm at 10 nm** steps (common in some rendering/color-science codebases and CIE-derived tabulations reduced to a coarse grid).
- **400–700 nm at 10 nm** (or up to 780 nm), also widely used, especially in reflectance-recovery/rendering literature (e.g., Jakob & Hanika 2019 and Meng et al. 2015 both work over the visible range at 5 or 10 nm resolution — see §7).
The official CIE data (CIE 018:2019 / CIE 015 colorimetry standard) tabulates the 1931 2° color-matching functions and the standard illuminants at 1 nm and 5 nm resolution over 360–830 nm (or 380–780 nm depending on table); 10 nm decimation is an accepted practical simplification. Source for CIE tabulated data: the official CIE data table pages — https://cie.co.at/datatable/cie-1931-colour-matching-functions-2-degree-observer — and the widely used non-CIE-affiliated CVRL (Colour and Vision Research Laboratories) tabulations at http://www.cvrl.org (per search-result summary; not independently refetched in this pass — flagged as "cite but not directly verified" since the WebFetch of cie.co.at was not attempted this session). CIE's own most current illuminant SPD dataset is published as CIE 2022, *"Relative spectral power distributions of CIE standard illuminants A, D65 and D50 (wavelengths in standard air)"*, DOI: 10.25039/CIE.DS.etgmuqt5 — available from the CIE Webshop (per search-result summary).

### 5.2 CIE 1931 2° color matching functions (CMFs)

The standard observer functions `x̄(λ), ȳ(λ), z̄(λ)` are the CIE 1931 2-degree standard colorimetric observer, defined by CIE and tabulated officially in CIE 018:2019 Table 6, and in the classic reference Wyszecki & Stiles, *Color Science: Concepts and Methods, Quantitative Data and Formulae* (1982), Table I(3.3.1) — per search-result citation. Practical machine-readable copies are commonly sourced from CVRL (http://www.cvrl.org) or bundled in open-source colorimetry libraries (e.g., the `colour-science` Python package) — not independently re-verified by direct fetch in this session; flagged as a minor gap (should verify numeric table directly before shipping in code).

### 5.3 Standard illuminants D65 and D50

- **D65** approximates average daytime/noon skylight and is the standard illuminant for sRGB colorimetry (white point `x=0.3127, y=0.3290` in CIE xy, giving XYZ white ≈ `(0.9505, 1.0000, 1.0890)` after normalizing Y=1) — per the Wikipedia sRGB fetch (below, §5.4) and standard practice.
- **D50** is the standard illuminant for graphic-arts/ICC printing workflows (used, e.g., in ICC profile PCS).
- Both are tabulated together with Illuminant A in the CIE 2022 dataset cited above.

### 5.4 XYZ from reflectance

Given illuminant SPD `E(λ)`, CMFs `x̄,ȳ,z̄(λ)`, and a sample reflectance `R(λ)`, the standard (Riemann-sum) computation is:

```
X = k · Σ_λ E(λ) R(λ) x̄(λ)
Y = k · Σ_λ E(λ) R(λ) ȳ(λ)
Z = k · Σ_λ E(λ) R(λ) z̄(λ)
```

with the normalizing constant `k = 100 / Σ_λ E(λ) ȳ(λ)` chosen so that a perfect reflecting diffuser (`R(λ)=1` everywhere) yields `Y=100` (this is the standard CIE colorimetry convention, referenced throughout colorimetry texts; not separately re-cited here beyond the general CIE 015 colorimetry standard framework already cited).

### 5.5 XYZ → sRGB (matrix + transfer function)

Fetched directly from the Wikipedia sRGB article (https://en.wikipedia.org/wiki/SRGB):

**Linear sRGB from XYZ (D65)** — the official inverse (XYZ→linear-RGB) matrix, IEC 61966-2-1 with the 2003 amendment's higher-precision coefficients:
```
[R]   [ 3.2406255  -1.5372080  -0.4986286] [X]
[G] = [-0.9689307   1.8757561   0.0415175] [Y]
[B]   [ 0.0557101  -0.2040211   1.0569959] [Z]
```
and the forward (linear-RGB→XYZ) matrix:
```
[X]   [0.4124  0.3576  0.1805] [R]
[Y] = [0.2126  0.7152  0.0722] [G]
[Z]   [0.0193  0.1192  0.9505] [B]
```
(both per the official 1999 IEC sRGB spec as summarized on the Wikipedia page; the D65 white point used is X=0.9505, Y=1.0, Z=1.0890).

**Transfer function (linear → encoded sRGB, "gamma"/OETF)**, piecewise linear-near-black + power law, with linear values `C_lin` and encoded `C_srgb`:
```
C_srgb = 12.92 · C_lin                         if C_lin <= 0.0031308
C_srgb = 1.055 · C_lin^(1/2.4) - 0.055          if C_lin >  0.0031308
```
and the inverse (decoding, encoded → linear):
```
C_lin = C_srgb / 12.92                          if C_srgb <= 0.04045
C_lin = ((C_srgb + 0.055) / 1.055)^2.4           if C_srgb >  0.04045
```
— both directions given in the Wikipedia sRGB article fetch (URL above), also consistent with the official spec at https://www.color.org/chardata/rgb/srgb.pdf (turned up in search but not independently refetched this session).

### 5.6 CIELAB

Standard CIE 1976 L\*a\*b\* from XYZ (normalized to a reference white `Xn, Yn, Zn`):
```
L* = 116 f(Y/Yn) - 16
a* = 500 [ f(X/Xn) - f(Y/Yn) ]
b* = 200 [ f(Y/Yn) - f(Z/Zn) ]
```
with
```
f(t) = t^(1/3)                    for t > (6/29)^3  (≈ 0.008856)
f(t) = t / (3 (6/29)^2) + 4/29     otherwise  (linear segment near black; equivalently written as 7.787t + 16/116)
```
— per standard colorimetry references, confirmed by search-result summaries citing the CIE 1976 definition and matching the well-known Bruce Lindbloom colorimetry reference pages (http://www.brucelindbloom.com/ — attempted direct fetch failed with a TLS handshake error this session; the formula above is the universally standard one and is corroborated by multiple independent secondary sources in the search results, e.g. Imatest's Color/Tone Appendix, https://www.imatest.com/docs/colortone_ref/).

### 5.7 ΔE color-difference formulas

- **CIE76** (`ΔE*ab`): plain Euclidean distance in L\*a\*b\* space —
  ```
  ΔE76 = sqrt[ (ΔL*)^2 + (Δa*)^2 + (Δb*)^2 ]
  ```
- **CIE94** (`ΔE94`): reweights by chroma to better match perceived differences —
  ```
  ΔE94 = sqrt[ (ΔL*/(kL·SL))^2 + (ΔC*ab/(kC·SC))^2 + (ΔH*ab/(kH·SH))^2 ]
  ```
  with `SL = 1`, `SC = 1 + 0.045·C*ab`, `SH = 1 + 0.015·C*ab` (reference conditions `kL=kC=kH=1`; the constant in `SC`/`SH` is sometimes given as 0.045/0.015 for graphic-arts applications vs. slightly different values, e.g. K1=0.048, for textiles) — per search-result synthesis citing standard CIE94 definitions, e.g. https://techkon.datacolor.com/cie-de-color-difference-equations/ and https://metricgate.com/docs/delta-e-ciede2000/.
- **CIEDE2000** (`ΔE00`): the most perceptually accurate and most complex, adding a lightness-dependent weighting `SL = 1 + 0.015(L̄'-50)² / sqrt[20+(L̄'-50)²]`, a chroma weighting `SC = 1 + 0.045·C̄'`, a hue weighting `SH`, a chroma-dependent a\* rescaling factor `G` (correcting CIELAB's non-uniformity for grayish/low-chroma colors), and a hue-rotation interaction term `RT` that specifically corrects known CIELAB distortion in the blue region — per search-result synthesis (https://techkon.datacolor.com/cie-de-color-difference-equations/, https://metricgate.com/docs/delta-e-ciede2000/, https://pkg.go.dev/github.com/jkl1337/go-chromath/deltae). With `kL=kC=kH=1`, a `ΔE00` near 1.0 is commonly treated as roughly a just-noticeable difference. *(Note: the full CIEDE2000 formula is intricate — dozens of terms — and was not reproduced in complete closed form from any single source fetched this session; implementers should pull the authoritative Sharma/Wu/Dalal 2005 paper or a vetted reference implementation rather than reconstruct it from these notes alone. This is flagged as a gap: exact CIEDE2000 constants should be double-checked against a primary source before coding.)*

---

## 6. The inverse problem: colorant formulation / recipe prediction

### 6.1 Classic approach — Allen's algorithm

E. Allen's papers in the 1960s–1980s are the foundational computer-color-matching (recipe prediction) references. Cited works include Allen 1966, 1974, and 1980; specifically "Basic Equations Used in Computer Color Matching, II. Tristimulus Match, Two-constant Theory," *JOSA* (1974) — per search-result summary (full text not independently fetched this session). The essence of Allen's method: given a target reflectance/tristimulus and a library of colorants each characterized by two-constant K-M coefficients, formulate and solve (originally via linear least-squares, restricting the practical number of colorants in any one recipe to ~3–4) for the mixture proportions whose predicted `K_mix/S_mix → R∞` best matches the target, either at the tristimulus level or spectrally. See summary discussion in Kubelka-Munk-or-neural-networks comparison paper — https://www.researchgate.net/publication/228945160_Kubelka-Munk_or_neural_networks_for_computer_colorant_formulation (also https://ui.adsabs.harvard.edu/abs/2002SPIE.4421..745W/abstract) and the review/derivation in Walowit, McCarthy & Berns, *"Spectrophotometric color matching based on two-constant Kubelka-Munk theory"*, Color Research & Application (1988) — https://onlinelibrary.wiley.com/doi/10.1002/col.5080130606 . A later refinement is Centore (2020), "Enforcing Kubelka–Munk constraints for opaque paints," https://onlinelibrary.wiley.com/doi/abs/10.1111/cote.12497, which explicitly reframes the fitting problem geometrically as finding the closest point on a **convex polytope** (the physically realizable region where all colorant concentrations are non-negative and K,S are physically non-negative) to a target, noting that naive ordinary-least-squares fits routinely produce physically impossible (negative or >1) concentrations that must be constrained away.

### 6.2 Spectral matching vs. tristimulus matching; metamerism

- **Tristimulus matching**: solve so that the predicted mixture's XYZ (under one chosen illuminant/observer) equals the target's XYZ. This produces, at best, a **metameric match** — visually identical under that one illuminant/observer combination but potentially divergent under others.
- **Spectral matching**: solve so that the predicted mixture's *entire reflectance curve* is close to the target's reflectance curve (e.g., minimizing sum-of-squared spectral differences), which if achieved closely enough gives a match that holds under (nearly) any illuminant — eliminating metamerism risk by construction. Paint-industry practice generally prefers spectral matching over tristimulus matching for exactly this reason — search-result synthesis of general color-matching literature (e.g., "color matches made in the paint industry are often aimed at achieving a spectral color match rather than just a tristimulus (metameric) color match" — summarizing patent/industry sources found in search, e.g. https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8456700).
- **Metamerism / index of metamerism**: the degree to which a tristimulus-only match diverges under a different illuminant, typically quantified as a ΔE (often ΔE\_CMC or similar) between the sample pair when both are re-evaluated under a second "test" illuminant — per search-result summary referencing Berns 1988 ("Quantification of illuminant metamerism for four coloration systems via metameric mismatch gamuts," Color Research & Application) — https://onlinelibrary.wiley.com/doi/abs/10.1002/col.5080130605 — and general treatment at https://en.wikipedia.org/wiki/Metamerism_(color).

### 6.3 Modern constrained-optimization / simplex formulation

Modern treatments explicitly frame recipe prediction as a **constrained optimization over the simplex**: colorant weights `c_i >= 0`, `Σ c_i = 1` (or `<= 1` if a "vehicle"/binder/medium fraction fills the remainder), minimizing some color-difference objective (spectral RMS, or ΔE76/94/2000 in Lab space) between the K-M-predicted mixture reflectance and the target. Centore (2020, cited above) makes this geometric framing explicit — the "physically realisable paint combinations" form a convex polytope and the correct fitting procedure is a constrained projection onto it rather than unconstrained least squares. Other modern approaches blend K-M with machine learning (gradient boosting, deep learning, elastic net) either as hybrids with or as replacements for the analytic K-M forward model, when K-M's assumptions (single-scattering, ideal diffuse geometry, homogeneous mixing) break down — see "Kubelka-Munk or neural networks for computer colorant formulation?" (search result, cited §6.1) and "Predicting Colour Reflectance with Gradient Boosting and Deep Learning," https://research.gold.ac.uk/id/eprint/35868/1/AIAI23.camera_paper140.pdf / https://link.springer.com/chapter/10.1007/978-3-031-34111-3_14. Linear-programming formulations specifically for textile dyeing recipe prediction (K-M + Duncan theory) are described in Moussa (2021), https://onlinelibrary.wiley.com/doi/abs/10.1002/col.22626.

---

## 7. Spectral upsampling: getting a plausible reflectance from an sRGB color

This is an **ill-posed inverse problem** (metamerism: infinitely many spectra map to the same tristimulus/RGB), so all methods below produce *a* plausible spectrum, not *the* true one.

- **Smits (1999)** — Brian Smits, *"An RGB-to-Spectrum Conversion for Reflectances,"* Journal of Graphics Tools 4(4), 11–22 (1999). Searches a metamer space for smooth, physically-plausible (bounded [0,1]) spectra reproducing a given RGB; a simple, efficient method historically used for texture upsampling in early spectral renderers. DOI 10.1080/10867651.1999.10487511. Reference implementation: https://github.com/colour-science/smits1999 .
- **Meng et al. (2015)** — Meng, Simon, Hanika & Dachsbacher, *"Physically Meaningful Rendering using Tristimulus Colours,"* Computer Graphics Forum 34(4) (EGSR 2015). Provides a fast method to compute a smooth reflectance spectrum from (almost) any XYZ input, distributing energy smoothly over wide wavelength bands like natural reflectances, addressing color shifts/energy-conservation violations that arise from naive RGB-space spectral rendering. Paper: https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.12676 , author copy: https://jo.dreggn.org/home/2015_spectrum.pdf .
- **Jakob & Hanika (2019)** — *"A Low-Dimensional Function Space for Efficient Spectral Upsampling,"* Computer Graphics Forum 38(2) (Eurographics 2019). Represents each reflectance as a smooth sigmoid-parametrized curve over 3 fitted coefficients per RGB, precomputed on a fine grid; claims the first method to achieve **zero error on the full sRGB gamut**, storage identical to RGB, and evaluation in as few as 6 FLOPs per wavelength; adopted in PBRT and Mitsuba. Paper: https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.13626 , author PDF: https://jo.dreggn.org/home/2019_sigmoid.pdf , reference implementation: https://github.com/mitsuba-renderer/rgb2spec , precomputed coefficient tables: https://zenodo.org/records/4050598 .
- **Otsu, Yamamoto & Hachisuka (2018)** — *"Reproducing Spectral Reflectances From Tristimulus Colours,"* Computer Graphics Forum 37 (2018), 370–381. Uses PCA on a corpus of measured spectra to derive a small set of basis functions, plus a greedy clustering scheme minimizing reconstruction error; converts tristimulus → spectrum at runtime via a single precomputed matrix multiply. Paper: https://onlinelibrary.wiley.com/doi/10.1111/cgf.13332 , author copy: https://cs.uwaterloo.ca/~thachisu/rgb2spec.pdf , project page: https://hi2p-perim.github.io/hotsu/project/rgb2spec/ .
- **Mallett & Yuksel (2019)** — *"Spectral Primary Decomposition for Rendering with sRGB Reflectance,"* Eurographics Symposium on Rendering (EGSR) 2019. Shows any sRGB reflectance can be expressed as a linear combination of three fixed "primary" spectra, one per BT.709 primary, such that under D65 illumination the combination reproduces the input sRGB triple exactly — an efficient, closed-form procedure (no per-color optimization at runtime) for generating a reflectance spectrum from sRGB. Paper: http://www.cemyuksel.com/research/papers/spectral_primary_decomposition.pdf , poster: https://geometrian.com/research/data/spectral-primaries/SpectralPrimaryDecompositionPoster.pdf , EG digital library entry: https://diglib.eg.org/items/bbffa865-e99c-4c1f-bd33-70102dc8af78 .

  **Note on naming/attribution**: the task description attributes a "spectral primary decomposition" method to Scott Burns, but the paper actually titled "Spectral Primary Decomposition..." is by **Mallett & Yuksel (2019)**, unrelated in authorship to Burns. Scott Burns's own distinct body of work (see next) is the **least-slope-squared family**, not "spectral primary decomposition" — this looks like a conflation in the task brief, and is flagged here explicitly rather than silently resolved.

- **Scott Burns — least-slope-squared reflectance recovery.** Burns has published a family of methods for reconstructing a smooth reflectance curve from a tristimulus or sRGB target by solving a constrained smoothness-minimization problem:
  - **LSS** (Least Slope Squared): minimizes the integral of the squared first derivative (slope) of the reflectance curve subject to reproducing the target tristimulus/RGB exactly; favors the flattest possible curve, but can produce **physically meaningless negative reflectance values**.
  - **LLSS** (Least Log-Slope Squared): performs the same minimization on `log(R(λ))` rather than `R(λ)` directly, which **guarantees strictly positive reflectance** — described as well suited to subtractive color-mixture simulation.
  - **LHTSS** (Least Hyperbolic-Tangent-Slope Squared) and an iterative variant **ILLSS**: further variants constraining reflectance to `[0,1]`; ILLSS reportedly "best matches paint and pigment colors found commercially and in nature" but at much higher computational cost.
  - Per search-result synthesis: "The curve that has the least sum of slope squared ... seems to match reasonably well the reflectance curves measured from real paints and pigments available commercially and in nature."
  - Primary sources: Burns, *"Generating Reflectance Curves from sRGB Triplets"*, arXiv:1710.05732 — https://arxiv.org/pdf/1710.05732 (also mirrored at http://scottburns.us/reflectance-curves-from-srgb/, direct fetch of that page failed this session with an expired-TLS-certificate error — flagged); Burns, *"Numerical Methods for Smoothest Reflectance Reconstruction"* (2020) — http://scottburns.us/wp-content/uploads/2025/03/Numerical-Methods-for-Smoothest-Reflectance-Reconstruction-Burns-2020.pdf ; MATLAB/Octave/Python source code release — http://scottburns.us/matlab-octave-and-python-source-code-for-refl-recon-chrom-adapt/ . Burns also has a separate, simpler graphics-oriented paper for approximate subtractive mixing without full spectral measurement, *"Subtractive Color Mixture Computation,"* arXiv:1710.06364 — https://arxiv.org/pdf/1710.06364 (fetched: uses spectral reconstruction of RGB values into "generic representative" spectra, then mixes via a weighted arithmetic-geometric mean — a K-M-adjacent but distinct, cheaper heuristic, explicitly positioned as an alternative to full K-M when spectrophotometric colorant data isn't available).

---

## 8. Reference material and datasets

### 8.1 Mixbox (Sochorová & Jamriška, 2021)

*"Practical Pigment Mixing for Digital Painting,"* ACM Transactions on Graphics 40(6) (SIGGRAPH Asia 2021), DOI 10.1145/3478513.3480549 — https://dl.acm.org/doi/10.1145/3478513.3480549 , author PDF: https://dcgi.fel.cvut.cz/wp-content/wpallimport-dist/publications/pdf/publications-2021-sochorova-tog-pigments-paper.pdf , project page: https://dcgi.fel.cvut.cz/en/publications/2021/sochorova-tog-pigments/ .

**How it differs from a straightforward K-M library**: rather than requiring users to supply per-pigment spectral K/S data and running K-M mixing directly, Mixbox precomputes (from real pigment measurements) a **latent color space** in which an ordinary RGB color is represented as a mixture of a small number of real, physically measured pigment primaries plus an additive "residual" term. Mixing is then done by linearly interpolating in this latent representation (which corresponds to mixing the underlying real pigments under K-M) and converting back to RGB — so K-M physics is "baked in" to a compact, fast, RGB-in/RGB-out API rather than exposed as explicit spectral math. This lets ordinary digital-painting tools get K-M-like mixing behavior (e.g., "blue + yellow → green" instead of the muddy gray of naive linear RGB mixing) without doing spectral rendering at all. Implementation (multi-language: C++, Python, JS, Unity, etc.), released under a CC BY-NC license: https://github.com/scrtwpns/mixbox (Python source excerpt fetched: `mixbox.py` at https://github.com/scrtwpns/mixbox/blob/master/python/mixbox.py).

### 8.2 Publicly available pigment/reflectance datasets

- **Pigments Checker "Modern & Contemporary Art" reflectance spectra database** (CHSOS / A. Cosentino) — an open-access, downloadable set of reflectance spectra for a checker/reference set of historical and modern artist pigments, intended for conservation science and imaging-method validation. Announcement/overview: https://chsopensource.org/chsos-application-note-4/ and https://chsopensource.org/pigments-checker-spectra-databases-on-spectragryph/ ; related publication A. Cosentino, *"FORS spectral database of historical pigments in different binders,"* e-conservation Journal 2, 57–68 (2014) (per search-result citation, PDF not independently fetched this session). The Checker's underlying Raman database is separately published: Cosentino, "Pigments Checker version 3.0, a handy set for conservation scientists: A free online Raman spectra database," Microchemical Journal 129, 123–132 (2016) — https://www.sciencedirect.com/science/article/abs/pii/S0026265X16301011 (Raman, not reflectance — noted for completeness, not directly usable for K-M work).
- **Artist Acrylic Paint Spectral, Colorimetric, and Image Dataset** — an IS&T Archiving Conference dataset paper documenting spectral reflectance of artist acrylic paints (masstones and tints) — https://library.imaging.org/admin/apis/public/api/ist/website/downloadArticle/archiving/19/1/10 ; related earlier work "Developing a Spectral and Colorimetric Database of Artist Paint" — https://www.researchgate.net/publication/36183327_Developing_a_Spectral_and_Colorimetric_Database_of_Artist_Paint (130 spectral reflectances of artist's paints reported per search-result summary). *(Direct download links / licensing terms were not independently verified this session — flagged as needing confirmation before relying on this as a data source; it's not clear from search results alone whether raw spectral CSVs are freely downloadable or only summarized in the paper.)*
- **No single, comprehensive, freely-licensed public database of two-constant K/S coefficients for artist pigments (as opposed to raw reflectance spectra) was found.** Reflectance-spectra datasets (Pigments Checker, the acrylic paint dataset) can in principle be converted to K/S via the standard remission-function transform (§1.2) if masstone + tint pairs are present, but this is a derived quantity, not something these sources publish directly as K,S tables. This is an explicit, confirmed gap: pigment K/S values used by proprietary color-matching software (e.g., commercial paint-industry systems referenced throughout §6) do not appear to be openly published.

---

## 9. Numerical pitfalls

### 9.1 Singularities in K/S at R=0 and R=1

From `K/S = (1-R)²/(2R)` (§1.2): as `R → 0`, `K/S → ∞` (division by zero) — a perfectly black/fully absorbing sample has "infinite" K/S under this transform, which is a real mathematical singularity, not just a numerical artifact. As `R → 1`, `K/S → 0` exactly (a perfect white/non-absorbing sample) — well-behaved, but reflectances measured or predicted to be *slightly above* 1 (which can happen from noise, fluorescence, or numerical error in a pipeline) make the numerator `(1-R)²` still positive but the physical interpretation breaks down (K/S should not go negative; a computed negative K/S indicates `R>1`, itself unphysical for a non-fluorescent, non-emitting sample). **Practical handling**: clamp measured/predicted `R` strictly into `(0, 1]`, typically to something like `[ε, 1]` with a small `ε` (e.g. `1e-4`–`1e-6`) rather than exactly 0, to avoid `inf`/`NaN` propagating through downstream computations (mixture solving, optimization gradients, etc.).

### 9.2 Why K/S is unbounded and what that means for mixing math

Because `K/S ∈ [0, ∞)` (unbounded above), *linear* combination of `K/S` values across colorants is **not** how K-M mixing actually works — mixing rules operate on `K` and `S` **separately** (§2.3: `K_mix = Σc_iK_i`, `S_mix = Σc_iS_i`), each of which is itself non-negative and (by the first law of thermodynamics, per Viggiano's paper) bounded only below by zero, not above — reflectance is what's bounded to `[0,1]`, not K or S individually. A common implementation mistake is to try to linearly interpolate/average `K/S` ratios directly instead of carrying `K` and `S` (or at least K and S in some fixed relative unit) separately through the mixing math; this produces incorrect mixtures except in the single-constant-with-shared-S special case (§2.2).

### 9.3 Negative values and other pathologies

- **Negative K or S from a naive linear inversion**: When solving the two-constant calibration equations (§4.1) from noisy real measurements, the resulting `K` or `S` per wavelength can come out numerically negative, which is unphysical (Viggiano's paper explicitly notes K, S, L, M, P "cannot be negative" by the first law of thermodynamics — https://s3.cad.rit.edu/cadgallery_production/documents/2083/KM_Pathology.pdf). Practical handling: clamp to zero, or use a constrained (non-negative-least-squares) fit rather than an unconstrained linear solve.
- **Negative reflectance from spectral-upsampling / reconstruction methods**: plain least-slope-squared (LSS, §7) reflectance recovery can yield negative reflectance values at some wavelengths — this is exactly why Burns introduced the log-domain (LLSS) and tanh-domain (LHTSS) variants that structurally enforce positivity (and boundedness to `[0,1]` for LHTSS) — per search-result synthesis of Burns's papers (§7).
- **Hyperbolic-cotangent / division-by-zero failures in the finite-thickness formula**: Viggiano's paper (directly extracted) identifies "attempting to take the hyperbolic cotangent of 0" and "attempting to divide by 0 or take the reciprocal of 0" as the two most frequent computational failures when implementing the classic K-M formulas (§1.4) on a computer, arising specifically when any of `b`, `S`, or `X` is zero (i.e., a non-scattering or zero-thickness layer) — see https://s3.cad.rit.edu/cadgallery_production/documents/2083/KM_Pathology.pdf, §2.1.2 and §2.2.2 of that paper (extracted directly). The paper's whole purpose is to derive reformulated, algebraically-equivalent "general case" formulas (avoiding special-cased branching) that are more robust to these edge cases under floating-point arithmetic, and to flag that classic exponential forms (§1.4, Eq. 1 in these notes) lose precision from repeated near-equal subtractions — another named pathology (catastrophic cancellation) distinct from outright division-by-zero.
- **Recommended general practice** (synthesizing the above): work in the hyperbolic or the "R0-with-sinh-multiplied-through" forms rather than the raw cotangent form where possible; clamp reflectance away from exact 0 and 1; use non-negative-least-squares or explicit simplex-projection (§6.3) rather than unconstrained least squares when fitting K, S, or mixture concentrations; and be aware that all of K, S, and any `K/S`-type ratio are **unitful** (inverse-length, tied to the assumed layer thickness or per-unit-concentration convention, §1.4) — mixing coefficients calibrated under one thickness/concentration convention are not directly comparable to another's without renormalization.

---

## Summary of explicit gaps / disagreements flagged above

1. Exact numeric convention for Saunderson K1/K2 varies across sources (0.04 vs 0.08 for K1; 0.4–0.6 for K2) — treat as measured/tunable, not fixed constants (§3.3).
2. No single canonical "tinting-strength" normalization standard was found for single-constant K-M in the art-conservation literature — appears to be handled ad hoc (§2.4).
3. Full closed-form CIEDE2000 formula was not reproduced verbatim from a primary source this session — verify against Sharma/Wu/Dalal 2005 or a vetted library before implementing (§5.7).
4. CIE CMF/illuminant numeric tables were identified by URL (cie.co.at, CVRL) but not independently refetched/verified this session (§5.1–5.2).
5. Scott Burns is *not* the author of the paper titled "Spectral Primary Decomposition for Rendering with sRGB Reflectance" (that's Mallett & Yuksel 2019); Burns's own distinct, differently-named methods are the LSS/LLSS/LHTSS family — the task brief appears to conflate these two bodies of work (§7).
6. No comprehensive, openly licensed public dataset of two-constant K/S coefficients for artist pigments was found; only raw reflectance-spectrum datasets (Pigments Checker, Artist Acrylic Paint dataset) exist publicly, from which K/S would have to be derived (§8.2).
7. Bruce Lindbloom's site (a commonly cited practitioner reference for exact CIELAB/matrix constants) could not be fetched directly this session due to a TLS handshake error — the CIELAB/matrix values used above were cross-confirmed from Wikipedia and other secondary sources instead, but a direct check against brucelindbloom.com is recommended before finalizing constants in code (§5.5–5.6).
