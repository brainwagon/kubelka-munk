# The model is always two-constant; scattering may be inferred from a single swatch

Paints are stored as separate absorption and scattering spectra and mixed as
`K_mix = Σ cᵢKᵢ`, `S_mix = Σ cᵢSᵢ`. Single-constant Kubelka-Munk, which keeps only the
ratio K/S, was rejected: it handles tints with white badly, and mixing with white is most
of what a painter does.

The difficulty is data. Two-constant calibration needs a masstone *and* a tint with a
known white per paint, and no free, redistributable K/S dataset for named artist paints
appears to exist — Mixbox exists precisely because that data is proprietary. So there are
two ways to make a Paint:

- `Paint.from_measurements(masstone, tint, white)` — the real calibration, and the
  first-class path.
- `Paint.from_srgb("#1F3A93")` — a convenience that recovers a plausible reflectance from
  one colour, inverts it to K/S, and *splits* that ratio into K and S using an assumed
  scattering profile.

The split in `from_srgb` is a documented fiction, not physics. It buys correct white and
tinting behaviour from a single hex code, and it is superseded the moment real
measurements are supplied. Both the docstring and the README must say so plainly rather
than let a caller assume the numbers are measured.
