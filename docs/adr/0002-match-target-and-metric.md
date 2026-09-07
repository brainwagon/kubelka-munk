# Matching minimises perceptual colour difference, not spectral difference

A target colour may be given as sRGB, as CIELAB, or as a measured reflectance spectrum;
all three are converted to one internal representation. The solver minimises CIELAB
colour difference under D65, so the answer is "looks the same to a human", not "is
physically the same curve".

CIEDE2000 is both the objective and the reported figure. An earlier version of this
decision used CIE76 as the objective -- on the grounds that it is smooth and better
behaved under gradient-based optimisation -- and reported CIEDE2000 on the result. That
was wrong, and measurement showed why: the two metrics do not merely differ in scale,
they disagree about which recipe is better.

Matching `#8a9a5b` from titanium white and yellow ochre, CIE76 is minimised at 65% white
and CIEDE2000 at 0% white. Optimising one while reporting the other therefore returned a
recipe that was not the best recipe by the number printed next to it, and made rankings
non-monotonic. Driving the search with CIEDE2000 directly costs about 30% more time and
measurably improves the matches (summed CIEDE2000 over a sample of eight targets fell
from 27.4 to 22.3); its hue-angle discontinuities did not trouble SLSQP in practice,
helped by the multi-start of ADR-0009.

CIE76 remains available by passing `objective=delta_e_1976` for a faster, cruder search.
Whatever objective is passed is also what selects the winning recipe, so the reported
figure is always the one that was minimised.

## Consequences

Matches are metameric: a recipe may match the target under D65 and drift under another
illuminant. Users who need illuminant-independence should minimise spectral RMS instead,
which the objective parameter allows.
