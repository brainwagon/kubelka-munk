# Handoff

State of this project as of commit `b84035c`. 19 commits, 144 tests passing, pushed to
https://github.com/brainwagon/kubelka-munk (public, `main`).

## What this is

Two things that grew out of one another.

A **library** (`src/kubelka_munk/`) that mixes artists' paints using Kubelka-Munk theory
— spectrally, over 36 wavelength bands — and solves the inverse problem of finding which
proportions of a palette come closest to a target colour. It depends on numpy and scipy
and nothing else.

A set of **example programs** (`examples/`) that use it, which is where most of the recent
work has gone. They repaint photographs with a limited palette of real paints, ending in a
painterly renderer that lays down brush strokes.

## Commands worth knowing

```sh
pip install -e ".[test,examples]"     # examples need Pillow and rembg
pytest                                 # 144 tests, ~40s

python examples/mixing_and_matching.py            # library tour: mix, and match a colour
python examples/zorn_wheel.py [rounds] [tints]    # colour wheel, writes zorn_wheel.svg
python examples/zorn_filter.py IMG [OUT] -m 12 --dither --blue
python examples/zorn_painting.py IMG [OUT] -t 0.875 --smoothness 0.7 \
       --separate --blue --background-tightness 0.5 --video OUT.mp4
```

That last line is the settings the paintings in this repo's history were made with. Around
40 seconds for a 1400px painting with video.

## Where things live

| | |
|---|---|
| `spectrum.py` | the wavelength grid, and reflectance sampled on it |
| `kubelka_munk.py` | the equations, and the Saunderson surface correction |
| `colorimetry.py` | spectrum → XYZ → sRGB, CIELAB, colour difference |
| `upsampling.py` | recovering a reflectance curve from one sRGB colour |
| `paint.py` | a Paint, and the two ways to build one |
| `palette.py` | mixing, and the search for a recipe |
| `palettes.py` | the shipped approximate palette |
| `examples/zorn_wheel.py` | the Zorn palette, calibrated; the wheel; `PRUSSIAN_BLUE` |
| `examples/zorn_filter.py` | flat per-pixel mapping, dithering, mixture selection |
| `examples/zorn_painting.py` | the painterly renderer — by far the largest file |

`CONTEXT.md` fixes the vocabulary. `docs/kubelka-munk-theory.md` is the research report,
with sources. `docs/adr/` records why the significant decisions went the way they did, and
is the first thing to read before changing any of them.

## Things that are true and non-obvious

Written down because each cost real time to discover, and none is visible from the code
alone.

**Saunderson's k₁ defaults to 0, not 0.04.** With sRGB-specified colours or
specular-excluded measurements, 0.04 counts surface reflection twice and imposes a floor:
nothing can appear darker than 4% reflectance, which turned every dark paint grey. See
ADR-0006.

**CIEDE2000 drives the matching search, not CIE76.** The two metrics disagree about which
recipe is *better*, not merely by how much — matching `#8a9a5b` from white and ochre, CIE76
is minimised at 65% white and CIEDE2000 at 0%. ADR-0002 records the reversal.

**A masstone alone does not determine how a paint mixes.** It fixes only the ratio K/S. Six
blacks with the identical masstone `#23201e` gave the same 3:1 mix with chroma from 4 to
32. This is why `Paint.from_measurements` (masstone *plus* a 1:9 tint) is the real path and
`from_srgb` is documented as an approximation.

**Two constants that must be judged locally, and one that must not.** The orientation
field's confidence gate is judged against the *local* neighbourhood, because a face is a
tenth the contrast of the buildings behind it and no global threshold serves both. The
`--smoothness` detail measure is judged *absolutely*, because relative measurement ranks a
uniformly-detailed region the same as a uniformly-flat one — it put spectacles below a bare
cheek. Getting these the wrong way round is easy and the results are plausible-looking.

**Blending an orientation towards a fixed angle rotates it.** The old field blended measured
tangents towards the hatching angle by confidence, which dragged half the canvas towards 40°.
Confidence now decides how readily a stroke *turns*, not which way it points.

**Floyd-Steinberg diverges on a gamut-limited palette.** A blue pixel leaves an error nothing
can correct; undiminished it compounds until the carried value is ±130 against colours in
[0,1]. The carried value is clamped at read time.

**A stencilled pass must start from its own ground.** The subject pass used to start from a
copy of the finished background and repaint only where it differed enough, so wherever the
sky came out close to skin — 16.9% of a forehead — the background's brushwork showed
through his face.

## Known limits, in rough order of how much they matter

- **The palette is not measured data.** Every paint is built from hex codes eyeballed off
  colour charts. No open dataset of Kubelka-Munk coefficients for artists' paints appears to
  exist. Treat the numbers as a plausible toy; `from_measurements` is there for real data.
- **Opaque films only.** No glazing or watercolour. The finite-thickness form is written out
  in the theory report but not implemented. ADR-0007.
- **Matches are metameric** — right under D65, possibly not under other light.
- `--smoothness` **retains** detail rather than sharpening it. Edge sharpness is bounded by
  the smallest brush, so genuinely sharper spectacles need a detail-adaptive finer layer,
  not a lower threshold.
- The tightness scale **compresses above about 0.85** — the brushes barely move. Use
  `--brushes` directly beyond that.
- The `--video` last frame is the finished painting but **not pixel-identical** to the PNG:
  H.264 at crf 20, median ΔE 1.28. The PNG is authoritative.

## Threads left open

- **Sharpen detail properly**: an extra, finer layer that only paints where detail is high.
  The most obviously missing thing.
- **More paint.** `--blue` halved the out-of-gamut pixels on both test photographs and made
  a green shirt green, because Prussian blue and yellow ochre mix to green. A green and a
  second red would be the next experiment; each needs a masstone and a 1:9 tint in
  `zorn_wheel.py`.
- **Finite-thickness Kubelka-Munk** for glazes. The representation already stores K and S
  separately, so this is an addition rather than a rewrite.
- **Anisotropic colour jitter** — generous in L\*, tight in a\*/b\*, so value can vary
  without hue wandering out of the flesh range. At jitter 10 the isotropic version reaches
  the palette's olives.

## Environment notes

- rembg 2.0.69 with models already cached in `~/.u2net`; ffmpeg 4.4.2 with libx264.
- A `python3 -m http.server 8001` was left running in the repo root to view outputs from a
  browser. **That will not survive this session** — restart it if wanted.
- Generated output (`zorn_wheel.svg`, `*_zorn*.png`, `*_painting*.png`, `*.mp4`) is
  gitignored; the repo is source only.
- Test images used were in `/mnt/c/Users/mvand/Downloads/` and are not in the repo.
