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
downstream.** Current architecture is right on this axis. This finding is now also
recorded in `docs/PIPELINE.md`'s fade restoration paragraph, as the reason the feature
composes upstream of Cast Removal's affine rather than running next to it.

## Finding 2, retracted: Variant A and B are not tied once the real detector runs

The original claim here — that a free-offset downstream affine absorbs an un-removed
stain exactly, making Option 1 (no upstream offset) and Option 2 (subtract a known stain
before the matrix) mathematically indistinguishable — was true only under the affine this
script fits: ordinary least squares directly against the true clean reference. That is an
**oracle affine**. It has no chroma gate, no pixel-selection step, nothing that a stain
could bias; it always finds the best possible fit by construction, so it cannot see the
one effect the ordering question was actually about.

Running the real detector settles it differently. `measure_neutral_axis_from_log` on a
stained fixture, compared against the same fixture with the stain removed first, selects
a visibly different, worse-conditioned population — the chroma gate reads part of the
stained region as neutral when it isn't. Feeding both through the full production
pipeline (`NormalizationProcessor` → `PhotometricProcessor`, real `neutral_axis_affine`,
not this script's stand-in) across four stain magnitudes gives:

```
stain magnitude    Variant A error (density)    Variant B error (density, oracle removal)
mild                       0.013                              0.006
moderate                   0.028                              0.006
heavy                      0.054                              0.006
severe                     0.091                              0.006
```

Two things about this table matter more than the ratio between its columns:

- **Read it in absolute density, not as a ratio.** 0.013 is invisible; 0.054 is marginal;
  0.091 is a real, visible channel imbalance. But all of it sits roughly two orders of
  magnitude below Finding 1's damage from Variant C (up to ~4 density). An un-removed
  stain biasing the neutral detector is real and monotone with stain severity — it is not
  the same *class* of problem as putting the fade matrix in the wrong place, and should
  not be weighed as if it were.
- **Variant B here used oracle stain removal** — the fixture's own known true stain value,
  subtracted exactly, not an estimate from a real detector. Nothing in the current design
  supplies that estimate: the rebate-probe idea is unverified and base yellowing is
  unmodeled. So 0.006 is a **ceiling under perfect knowledge of the stain**, not a
  forecast for a buildable Option 2 estimator. The honest statement of this finding is:
  *an upstream offset is worth up to ~0.09 density on severe stain, given exact knowledge
  of the stain* — a bound on the prize, claimable only if the offset can actually be
  measured.

## How good would that estimate need to be

Perturbing the injected offset by a fractional error `eps` (`estimated = true_stain * (1
- eps)`, `eps=0` is the oracle above, `eps=1` is no correction at all, i.e. Variant A) on
the heavy-stain fixture through the same real pipeline:

```
eps    error (density)
0.0    0.006   (oracle)
0.5    0.022
0.9    0.047
1.0    0.054   (= Variant A, no correction)
1.02   0.054   (already worse than Variant A -- exceeds it just past this point)
2.0    0.119
```

The result is asymmetric and forgiving in the direction that matters: any estimate that
removes *some* of the stain in the right direction — even a small, conservative fraction
of it — helps monotonically, all the way down from the oracle to no correction at all.
The failure mode is overshoot: once the residual left behind (`true − estimated`) exceeds
the original stain's own magnitude, which happens the moment the estimate passes the true
value and flips sign, the correction is actively worse than doing nothing, and the margin
past that point is thin (~2% overshoot is enough to lose the benefit on this fixture). A
sign-flipped estimate is roughly twice as bad as no correction at all.

This sizes the requirement for a real probe: it does not need to be precise, but it does
need to reliably get the *direction* right and not overshoot the *magnitude* — a much
lower bar than matching the oracle, but a real one, and one nothing in the current design
(no verified rebate probe, no base-yellowing model) currently meets.

## Process note: oracle-based validation hides selection effects

This is the third correction in this line of work traceable to the same root cause: a
ground-truth- or oracle-based validation methodology structurally cannot see the effect
it is meant to rule out, because it has no selection step of its own to be biased. The
pre-fix survival-ratio estimator's own composed unmix papered over delta contamination
the same way; here, a least-squares affine fit directly against the known clean image had
no chroma gate to fool, so it could not see a chroma gate being fooled. Worth keeping in
mind before trusting an oracle-fitted synthetic result on anything that depends on real
detector logic making a selection: fit the actual selection step, not a stand-in for its
best possible outcome, or the result answers a different question than the one being
asked.

## Recommendation

- **Don't build Variant C.** The synthetic evidence is one-sided and matches the
  plan's own prior; reordering Cast Removal is not worth proposing upstream.
- **Don't build Option 2 (an upstream stain offset) yet.** The prize is real but small
  relative to Finding 1, and is bounded by a test that assumed perfect knowledge of the
  stain. Nothing in the current design supplies that knowledge. Building the offset
  mechanism ahead of a way to estimate its input would be building the cheap half of a
  feature whose expensive half — a real stain estimator, accurate enough on direction and
  magnitude per the sweep above — does not exist yet.
- **The real gap is a stain estimator, not the offset plumbing.** If one becomes
  available (a verified rebate probe, a base-yellowing model, or real faded-slide
  material to characterize either against), this experiment's ε-sweep is the test to
  re-run against its actual accuracy, not the oracle case.
- The real-slide validation gap is separate from all of the above: none of this has been
  checked against an actual faded slide, only synthetic fixtures. Worth asking on Photrio
  for a faded Agfachrome or 1970s-80s Ektachrome scan if nothing suitable turns up
  locally — one file is enough to start.
