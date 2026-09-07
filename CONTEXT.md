# Kubelka-Munk Paint Mixing

A library for mixing artists' paints the way pigments actually behave, and for finding
which proportions of a given palette come closest to a wanted colour.

## Language

**Paint**:
A named colorant as it comes out of a tube, described by its absorption and scattering
across the visible spectrum.
_Avoid_: Pigment (properly the raw powder, not the tube), colour, ink

**Palette**:
The ordered set of Paints a painter is willing to draw on for a given mix.
_Avoid_: Swatch set, colour library

**Mixture**:
The forward direction — a set of Paints combined in stated proportions, and the Spectrum
that results.
_Avoid_: Blend, composite

**Recipe**:
The inverse direction — the proportions a search *found* for a Target, together with how
close it got. Distinct from a Mixture: a Mixture is what you asked for, a Recipe is what
was discovered.
_Avoid_: Solution, formula, match

**Target**:
The colour a Recipe is trying to reach, however it was originally specified.

**Spectrum**:
Reflectance sampled at regular wavelengths across the visible range.
_Avoid_: Curve, SPD (a spectral power distribution is a property of a light source, not of
a surface)

**Absorption** (K):
How strongly a Paint takes light out of the beam at each wavelength.

**Scattering** (S):
How strongly a Paint turns light back at each wavelength. What makes a white paint white,
and what gives it its tinting strength.

**Masstone**:
The colour of a Paint applied thickly and unmixed.
_Avoid_: Pure colour, base colour

**Tint**:
A Paint mixed with a known white, in a known proportion. Together with the Masstone this
is what lets absorption and scattering be told apart.

**Tinting strength**:
How far a Paint carries in a mixture -- how much of it you need before it makes itself
felt against white. A property of scattering, not of colour: two Paints can look alike
from the tube and differ entirely here.

**Metamer**:
A colour that matches another under one light and parts company under a different one.
What a perceptual match produces, and the reason a match is reported against a stated
illuminant.

**Saunderson correction**:
The adjustment between reflectance as an instrument measures it and the internal
reflectance the Kubelka-Munk equations describe, accounting for light bouncing off the
paint's surface without ever meeting a pigment.
