# kubelka-munk

Mixing artists' paints the way pigments actually behave, and working backwards from a
colour to the proportions that make it.

```python
from kubelka_munk import APPROXIMATE_ARTIST_PALETTE as palette

palette.mix({"Ultramarine Blue": 1, "Cadmium Yellow Light": 3})
# <Mixture #5b7f4c = Ultramarine Blue 25%, Cadmium Yellow Light 75%>

palette.match("#8a9a5b", max_paints=3)
# <Recipe #899a5a dE 0.34 = Titanium White 21%, Yellow Ochre 77%, Phthalo Blue 2%>
```

Blue and yellow give green. Blending the two colours arithmetically gives grey-brown mud,
which is why digital colour pickers feel wrong to anyone who paints.

| yellow | as paint | as pixels |
|-------:|:---------|:----------|
| 0%     | `#2b3f9e` | `#2b3f9e` |
| 25%    | `#275766` | `#606476` |
| 50%    | `#3b6757` | `#95894f` |
| 75%    | `#5b7f4c` | `#caae28` |
| 100%   | `#ffd300` | `#ffd300` |

## How it works

Each paint is described not by a colour but by how strongly it absorbs (`K`) and scatters
(`S`) light at each of 36 wavelengths across the visible spectrum. Kubelka-Munk theory
says these quantities simply add in proportion when paints are mixed, and gives a formula
for the colour that results. Green appears because blue absorbs the long wavelengths and
yellow absorbs the short ones, leaving the middle of the spectrum — which is what green
is — for neither of them to take away.

Going the other way, `match` searches for the proportions whose mixed colour comes closest
to a target, minimising CIEDE2000 colour difference.

## Installing

```sh
pip install -e ".[test]"
pytest
python examples/mixing_and_matching.py
```

Requires Python 3.10+, numpy and scipy.

## Reading it

- `spectrum.py` — the wavelength grid, and reflectance sampled on it
- `kubelka_munk.py` — the equations themselves, and the Saunderson surface correction
- `colorimetry.py` — spectrum to XYZ to sRGB, CIELAB, and colour difference
- `upsampling.py` — recovering a plausible reflectance curve from a single colour
- `paint.py` — a Paint, and the two ways to build one
- `palette.py` — mixing, and the search for a recipe
- `palettes.py` — the ready-made palette

`CONTEXT.md` defines the vocabulary, `docs/kubelka-munk-theory.md` covers the theory and
the sources, and `docs/adr/` records why each significant decision went the way it did.

## What this does not do, and where it is guessing

These limits are deliberate and worth knowing before trusting a number.

**The shipped palette is not measured data.** `APPROXIMATE_ARTIST_PALETTE` carries the
names of real artists' colours, but each entry is built from a hex code eyeballed off a
colour chart and a tinting strength set by judgement. No open dataset of Kubelka-Munk
coefficients for artists' paints appears to exist — the values behind commercial
colour-matching systems are proprietary. Treat the palette as a plausible toy.

**`Paint.from_srgb` invents information.** A single colour cannot tell you how a paint
behaves in a mixture, because that depends on how much it scatters, and two paints can
look identical from the tube and behave completely differently once white is added. This
constructor reconstructs a plausible reflectance curve from the colour, then splits the
resulting `K/S` into `K` and `S` by *assuming* scattering is flat across wavelength and
equal to `tinting_strength`. Paints built this way are flagged `is_approximate`. Set
`tinting_strength` deliberately — high for an opaque cadmium, low for a transparent
quinacridone — or every paint on your palette will tint alike, which no real palette does.

For real work, measure your paints and use `Paint.from_measurements`, which performs the
standard two-constant calibration from a masstone and a tint.

**Matches are metameric.** A recipe matches its target under D65 daylight. Under a
different light the two may part company. Pass a spectral objective to `match` if you need
a match that holds under any illuminant.

**Only opaque films.** Paint layers are treated as thick enough to hide whatever is
underneath. Glazing, scumbling and watercolour need the finite-thickness form of the
theory, which is not implemented — see `docs/adr/0007`.

**Kubelka-Munk is itself an approximation.** It assumes light travels in just two
directions through a uniform medium. Real paint has particles of varying size, binder that
scatters on its own account, and pigments that misbehave; the model is a good working
account of mixing, not physics.
