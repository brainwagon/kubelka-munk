# Paints are represented as sampled reflectance spectra, not RGB triples

Kubelka-Munk is a per-wavelength law. We sample the visible range at 10nm and run all
mixing math per band, converting to XYZ/sRGB only at the boundary for display and for
colour-difference metrics.

The tempting alternative — running K-M over three RGB channels, as much "K-M in a shader"
code does — is rejected. Three channels cannot reproduce the hue shift that makes
blue + yellow give green, which is the entire reason to prefer K-M over a linear blend.
The cost is that every paint needs a full spectrum, which pushes complexity into how
pigment data is obtained (see ADR-0005).
