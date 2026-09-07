# Version 1 models opaque films only

Paint layers are treated as optically infinite: whatever is underneath does not show
through. The finite-thickness Kubelka-Munk solution — the one with the hyperbolic
functions and a substrate reflectance — is what glazing, scumbling and watercolour need,
and it roughly doubles the surface area of the physics while adding a "what is underneath"
parameter to every call.

This is a scope boundary stated in the README, not a silent assumption. Storing K and S
separately is already the representation a finite-thickness extension needs, so v2 is an
addition rather than a rewrite.
