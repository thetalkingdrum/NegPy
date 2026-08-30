# Results: fade/Cast Removal placement experiment

Run with `uv run python experiments/fade_placement/run_experiment.py`. Numbers below
from that run; regenerate rather than trust this file if the script changes.

```
fixture                          variant                 max err   mean err  affine moved
-----------------------------------------------------------------------------------------
control (no fade, no stain)      A baseline              0.00000    0.00000       0.00000
control (no fade, no stain)      B upstream offset       0.00000    0.00000       0.00000
control (no fade, no stain)      C fade downstream       0.00000    0.00000       0.00000
mild fade, no stain              A baseline              0.00000    0.00000       0.00000
mild fade, no stain              B upstream offset       0.00000    0.00000       0.00000
mild fade, no stain              C fade downstream       0.40081    0.12414       0.13178
heavy fade, no stain             A baseline              0.00000    0.00000       0.00000
heavy fade, no stain             B upstream offset       0.00000    0.00000       0.00000
heavy fade, no stain             C fade downstream       4.17481    1.23953       1.03061
mild fade, heavy stain           A baseline              0.00000    0.00000       0.15012
mild fade, heavy stain           B upstream offset       0.00000    0.00000       0.00000
mild fade, heavy stain           C fade downstream       0.39776    0.12421       0.28305
differential fade, with stain    A baseline              0.00000    0.00000       0.08437
differential fade, with stain    B upstream offset       0.00000    0.00000       0.00000
differential fade, with stain    C fade downstream       4.08690    0.89476       1.34686
```

Control fixture passes through unchanged on every variant (max err ~6.66e-15) — the
sanity check Step 3 requires.

## Finding 1: Variant C is clearly worse

Fitting the affine to the *raw faded* capture before the matrix, instead of after, does
real damage — max error 0.40–4.17 in density, worst exactly where it matters most
(heavy fade). The raw capture is still cross-channel mixed; a per-channel affine
(gain + offset, no off-diagonal term) cannot separate what's genuinely affine-correctable
from what's a matrix (cross-channel) problem, so it fits a wrong affine that then gets
baked in before the matrix restore, compounding rather than helping.

This confirms the plan's own suspicion ("metering the corrected film is arguably the
right thing") and answers the placement question directly: **don't move the fade matrix
downstream.** Current architecture is right on this axis.

## Finding 2: Variant A and B are mathematically indistinguishable in final accuracy

Both hit exact recovery (0 to float precision) on every fixture, stain or no stain. This
is provable, not a coincidence of these fixtures: `F⁻¹(D − o) − [F⁻¹(D) − o] = o − F⁻¹(o)`
— a **constant** per-channel difference, independent of the image content. Any downstream
affine with a free offset term (Cast Removal's `neutral_axis_affine` has one) absorbs a
constant exactly, by construction. So Task B's ordering concern — "physically it should be
`F⁻¹(D − o)`, not `F⁻¹(D) − o`" — is true as stated, but the error it describes is
*exactly* the kind of error Cast Removal already exists to remove, provided Cast Removal
runs on every frame that needs it.

The **affine moved** column shows the real, non-cosmetic difference: on the stained
fixtures, Variant A's affine has to move (0.084–0.150) to clean up the stain; Variant B's
doesn't (0.000, since the stain was already gone before the matrix). Same final image,
different *division of labor* between the physically-motivated stage and the cheap one.

## Caveat: this uses a best-possible affine, not the real detector

Each variant's affine is fit by ordinary least squares directly against the true clean
reference — the best any affine could do, not what `measure_neutral_axis_from_log`'s
chroma-gated neutral selection would actually find. This was deliberate, to isolate the
placement question from detector quality (a separate, already-tested concern). But it
means Finding 2 has a real blind spot: if an un-removed stain shifts which pixels a *real*
detector selects as neutral (its chroma gate could read a stained patch as less neutral
than it is, or vice versa), Variant A and B could diverge in practice even though they're
provably identical under a perfect affine. That would need the actual
`measure_neutral_axis_from_log` / `neutral_axis_affine` machinery run against these
fixtures, not this script's stand-in, to settle.

## Recommendation

- **Don't build Variant C.** The synthetic evidence is one-sided and matches the
  plan's own prior; reordering Cast Removal is not worth proposing upstream.
- **Task B (Option 1 vs 2) is close to moot for final image quality**, provided Cast
  Removal is on. The real question left is whether an un-removed stain measurably biases
  the *real* neutral detector's pixel selection — untested here, and the one thing worth
  checking with the actual detector (or a real stained slide) before closing Task B for
  good, per the plan's own "one test image" suggestion.
