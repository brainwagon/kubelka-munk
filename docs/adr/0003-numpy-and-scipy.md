# numpy and scipy are dependencies; legibility comes from naming, not from asceticism

The stated goal is maximum legibility. We read that as "the code looks like the equations
in the paper", which numpy delivers and hand-rolled loops obscure. scipy supplies the
constrained solver over the simplex.

Rejected: a pure-stdlib implementation. It would make every spectral operation an explicit
loop and force us to hand-roll an optimiser, trading a small dependency list for a large
amount of incidental code. Quantities are named after the literature (`absorption`,
`scattering`, `K`, `S`) rather than compressed into one-letter soup.
