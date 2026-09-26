# Fade restoration profile gallery

Community-contributed dye-fade parameters for NegPy's **Fade Restoration**
control (Process panel, E-6 only).

Every `.toml` here is bundled with the app and shows up, read-only, in the
sidebar dropdown. Your own profiles live in `<Documents>/NegPy/fade/`.

A profile is `delta`: the six side-absorption ratios between layers, in
`(gr, br, rg, bg, rb, gb)` order — a property of the dye set and the capture,
not of any one faded frame. A measured or spec-sheet profile also records the
`bands` it was read at, the R/G/B wavelengths in nm (see below); a profile
tuned by eye on a rig has none. A malformed `bands` is rejected, the same as a
malformed `delta`. The two surviving-dye ratios that *do* vary per
frame (relative green/blue survival against red) are not profile data; they
live as sliders in the sidebar, or from the Estimate action, next to
Strength.

```toml
process = "Transparency"   # or "Color Negative", once a negative dye set exists
bands = [650, 550, 450]    # R, G, B wavelengths, nm; omitted for a tuned profile
```

## Where these numbers come from

`delta` here is **measurement side absorption** — a dye's unwanted density in
a band that isn't its own, at the specific wavelengths a channel measures it
at. It is a genuinely different quantity from *interlayer fade coupling* (how
much one layer's own fading rate depends on its neighbours), which shares the
same 3×3 algebraic shape but has different physical origins and is not
modelled here yet.

Every `450/550/650` file is computed from spectral dye-density curves
digitised by [`JanLohse/spectral_film_lut`](https://github.com/JanLohse/spectral_film_lut)
(MIT), which traced the manufacturers' own published datasheet plots. Values
were cross-checked against an independent digitisation in
[`andreavolpato/spektrafilm`](https://github.com/andreavolpato/spektrafilm)
(profiles CC BY-SA 4.0 — used for verification only; nothing shipped here is
derived from it) and agree to about 0.005 on every shared stock.

**These numbers are specific to a 450/550/650 nm narrowband scanner**
(Gschwind's canonical bands, ~20 nm half-width interference filters). Moving
the red band to a broadband/colorimetric sensor's response peak (~590 nm)
changes green's leak into red by roughly an order of magnitude on the stocks
checked — a fade profile is meaningless without knowing the scanner's channel
wavelengths, exactly as NegPy already treats Crosstalk profiles as belonging
to a whole scanning setup, not the film alone.

**On a broadband light** (a white LED, a flash, a flatbed lamp) the side
absorption is much larger and depends on the light and the camera, so no
bundled number describes it. Copy a bundled profile or start a new one in the
editor, then tune the six terms by eye on a frame with known neutrals, the
same way a Crosstalk profile is tuned for a negative. It saves as
`type = "tuned"`, with no `bands`.

`Generic E6` is the mean of Ektachrome 100D, Provia 100F and Velvia 50 — a
reasonable default, not a specific stock's numbers. Per-stock profiles differ
enough to be worth selecting directly when you know the film: Kodachrome 64's
`rg` (cyan's leak into green) is roughly a quarter of the E-6 stocks', a real
difference in dye chemistry, not noise.

Measured profiles (fit against real faded and unfaded scans of the same
stock) remain the most useful contribution beyond this: they'd capture
interlayer coupling and base staining that a pure spectral-density read does
not. A stained or yellowed base in particular is an additive density floor,
not a side-absorption ratio, and no matrix here will remove it — a known,
unmodelled gap, not an oversight in these numbers.
