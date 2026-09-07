# The Saunderson correction is applied, at exactly two boundaries

Paint has a glossy surface: some light reflects off the front without meeting a pigment,
and some is reflected back down from the inside. Ignoring this is a well-known error
source that makes dark mixtures come out too light, so the correction is applied with the
usual defaults (k₁ = 0.04 external, k₂ = 0.6 internal), settable per Paint, and disabled
by setting both to zero.

It is applied in exactly two places — measured reflectance to internal on the way in, and
internal back to measured on the way out — because applying it twice, or in the wrong
direction, is the obvious bug. A unit test pins the round trip.
