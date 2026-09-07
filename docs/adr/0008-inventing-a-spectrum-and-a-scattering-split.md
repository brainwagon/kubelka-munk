# `from_srgb` uses least-hyperbolic-tangent-squared upsampling and a flat scattering profile

Building a Paint from one hex code requires two inventions, and this records both so a
future reader does not mistake either for measurement.

**Spectrum from colour.** Infinitely many reflectance curves produce the same sRGB. We use
Burns's least-hyperbolic-tangent-squared (LHTSS) method: the smoothest curve that
reproduces the target colour exactly, with the tanh substitution keeping the result inside
[0, 1] without clamping. Chosen over Smits (1999), Meng (2015) and Jakob & Hanika (2019)
because it ships no data table and its derivation is short enough to state in a docstring,
while smoothness is a genuine property of real paint reflectances.

**Splitting K/S into K and S.** Scattering is taken as flat across wavelength and equal to
a per-Paint `tinting_strength` (default 1.0); absorption follows as `K = (K/S) · S`. A
wavelength-dependent profile would be more physical but adds a parameter with nothing to
calibrate it against. One clearly-labelled dial captures what actually matters in
practice — that a chalky opaque paint and a transparent glazing colour behave differently
when mixed with white — and setting it to 1.0 everywhere recovers the naive assumption.
