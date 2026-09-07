# A recipe is a point on the simplex; palette subsets are searched by brute force

Recipe weights are non-negative and sum to one — they partition a fixed volume of paint,
which is what a painter actually mixes. White is an ordinary palette entry with no special
casing; K-M already gives it its tinting behaviour through its large scattering term.

A "use at most N paints" constraint is honoured by enumerating subsets and solving each.
Palettes are small (a dozen paints, C(12,3) = 220 solves), so exhaustive search is cheap
and exact, and avoids the tuning burden of a sparsity penalty.

Out-of-gamut targets are not an error condition: the achieved colour difference is always
returned on the result, so "unreachable" is a number the caller inspects rather than an
exception they must catch.
