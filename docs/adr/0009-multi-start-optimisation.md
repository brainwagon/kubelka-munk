# Matching is multi-start SLSQP with a fixed seed

The objective is not convex in the recipe weights, so a single SLSQP solve from the
simplex centroid lands in a local minimum often enough to matter. We solve from the
centroid plus a number of seeded-random simplex points and keep the best.

The seed is exposed and fixed by default, so the same target and palette always give the
same recipe — reproducibility is worth more here than the last fraction of a delta-E.
A global optimiser such as differential evolution was rejected as far slower for no
practical gain, since the `max_paints` subset search already decomposes the problem into
small, well-conditioned subproblems.
