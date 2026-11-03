# Fade Restoration crossover table and NegativeConverter A/B: results

Branch `experiment/fade-crossover-table`, from `claude/raw-density-order`. CPU and GPU for the table, the scanner zero and the fog bake; CPU only for the NegativeConverter stages. Defaults render bit-identically to the base branch.

Reproduce the numbers:

```bash
uv run python experiments/fade_crossover/evaluate.py      # phases 1 and 2
uv run python experiments/fade_crossover/phase3.py      # phase 3: scanner zero, tone, fog
uv run python experiments/fade_crossover/phase4.py      # phase 4: gray finders, tone, fog diagnostics
```

## Phase 4

Every new setting is on the `fade` roll card (in `LATE_FADE_DEFAULTS`), in the Fade Restoration paste row and the card reset, and translated in all four catalogs. No field was renamed, so `migrations.py` has no entry. The defaults (`fade_neutrals_mix` 0, `fade_tone_strength` 1, Neutrals NegPy, Tone Off, fog Off) render bit-identically: Mix and No Sky act only when Estimate runs, Tone Strength only when a Tone that writes the sliders is chosen, and the fog refactor keeps the bake's arithmetic (`test_a_clean_scan_stays_bit_identical`, `test_the_defaults_render_as_before`). Nothing new reaches the GPU: Estimate and the tone fits are CPU analysis that write settings, and the fog bake is a source bake ahead of both engines. `THUMBNAIL_RENDER_VERSION` is not bumped (the base branch already bumped it over main, and no default render changes).

Reproduce: `uv run python experiments/fade_crossover/phase4.py [neutrals] [tone] [fog]`.

### 1. Neutrals Mix (`fade_neutrals_mix`)

A **Mix** slider (0 to 100 %) under Neutrals in the Compare fold. Estimate reads each gray ref (midtone, shadow, highlight; log, per channel) as `(1 − m)·r + m·r_nc`, before the survivals, the lean and the table fit (`fade.gray_refs`). Mix 0 is NegPy's reader exactly and Mix 100 % is NC's exactly (`test_mix_ends_are_the_two_readers_exactly`). In between, a frame where one reader finds no grays takes the other's. Mix acts under Neutrals NegPy and No Sky, is grayed out under NC, and shows only while Color model is NegPy. Like Neutrals, it changes nothing until Estimate runs again; the tooltip says so.

### 2. The cold lean

Method: a NumPy copy of NegPy's `neutral_axis` with each suspect switchable (`experiments/fade_crossover/neutral_finders.py`; `check_replica` asserts it matches `negpy_core` on every Coolscan-like fixture). New scenes in `tests/fade_crossover_scene.warm_scene`, each with a true gray ramp over the top 10 % of the frame:
- **wall**: a large wall lit to shaded, sand, skin, some sky and foliage, the wall and sand warm by `warmth` × (0.03, 0, −0.05) and (0.04, −0.02, −0.14) log E;
- **sun**: gray objects in warm sun on one half and cool skylit shade on the other, the ramp only over the midtones;
- **bright**: the wall layout with a warm sunlit area as the frame's brightest part;
- **sky**: the wall layout under a saturated blue sky over the top 40 %.

Each reader runs a full Estimate (Midtone); the score is the render's mean b\* over the true gray ramp's midtone steps (negative = cold), Coolscan-like and Plustek-like (broadband mix, white 1.3 stops down), linear fade 1/.75/.55 and the crossover fade.

**NC's reader is not warmer than ours on these fixtures; it is colder.** Over all 32 fixture renders at Crossover 100 %:

| reader | mean b\* | mean \|b\*\| | shift from NegPy |
|---|---|---|---|
| NegPy | −2.67 | 3.52 | 0 |
| NegPy, Mix 50 % | −4.08 | 4.68 | −1.41 |
| NC | −3.90 | 4.33 | −1.23 |

Mix 50 % is colder than either end: a midtone ref halfway between two readers that sit at different densities fits survivals that neither reader gives. NC's mean-shift settles each band on its largest near-neutral cluster, so a large uniform warm area (skin, a wall) wins its band. On the wall layout at warmth 1, NC's midtone bands read b\* +8.8 and +3.4 in the unfaded film, NegPy's midtone +3.4 (`phase4.py`, band table).

**The one case that reproduces "ours is colder" is sun and shade on the Plustek-like scan**: NegPy −2.0 and −4.2 b\* where NC reads −0.1 and +0.4 (crossover fade, warmth 1 and 2). The cause is the **sky sector**: in the frame's own per-channel stretch, skylit shade reads blue enough (chroma above 0.06, hue −150° to −60°) to be dropped as sky, 19 to 23 % of the cells. That leaves the warm sunlit grays as the neutral, and the render goes cold. With the foliage sector only, the same frames read +1.3 and +2.5.

Each suspect alone, mean over all fixtures at Crossover 100 % (0):

| suspect, alone | mean \|b\*\| | shift in mean b\* |
|---|---|---|
| none (NegPy) | 3.52 (4.21) | 0 |
| no chroma caps | 3.52 (4.21) | 0.00: the caps never bind on these frames |
| no subject sectors at all | 3.67 (4.43) | +0.05 (+0.13): the foliage sector helps where the sky sector hurts |
| **sky sector off** (= No Sky, the shipped reader) | **3.06 (3.89)** | **+0.69 (+0.72)** |
| soft weight 0.25 D (or 0.12 D) instead of the 30 % quantile | 3.34 (3.76) | −0.02 (+0.32) |
| 8 bands, NC's layout | 3.32 (5.98) | +0.04 (−1.95) |
| soft weight and 8 bands | 3.62 (4.93) | −0.29 (−0.83) |
| raw-log chroma instead of the stretch | 3.71 (4.72) | −0.23 (−0.61) |

**Sky sector off wins**, and it is the only change that moves the lean warm. It is better even on the saturated-sky layout (Coolscan linear −7.6 → −2.1, crossover −10.7 → −6.9), because a saturated sky never enters the lowest-chroma set; the sector instead drops grays the fade's cast pushed toward blue. It is worse on one fixture: Coolscan crossover, bright layout, −9.5 → −10.9.

It ships as a third Neutrals choice, **No Sky** (`fade_ab_neutrals = "no_sky"`): `neutral_axis_from_grid(grid, sky_sector=False)`, which passes an infinite chroma floor for the sky sector to `negpy_core.neutral_axis` (no Rust change; every fallback pass keeps the setting). The default reader is unchanged.

### 3. Tone Strength and Adaptive

**Tone Strength** (`fade_tone_strength`, 0 to 100 %, default 100 %) shows while Tone is NegPy or Adaptive. Each slider gets `v0 + t·(v1 − v0)`, rounded to its step: `v0` is what the slider prints when the Tone is chosen (an Auto Density or Auto Grade's metered value included, `transfer_shown_values`), `v1` the fit. A strength change writes the sliders again from `fade_tone_saved`, so Off still restores them exactly (`test_strength_moves_the_sliders_from_the_saved_values_and_off_still_restores`). Under Off or NC the slider is hidden and the value is only stored.

**Adaptive** (`fade_ab_tone = "adaptive"`, `fade_tone.adaptive_values`) does not use their curve. It reads the same `fade_tone_points` (the frame's mean density after the fade and the table, through the slide window) and solves the two sliders directly:

| constant | value | why |
|---|---|---|
| `ADAPTIVE_SPAN` | 2.4 D | the print density span the frame's 0.5 to 99.5 % span maps to: about what a well-exposed slide's picture holds |
| `ADAPTIVE_CONTRAST` | 1.0 to 1.5 (ISO-R 115 to 77) | the floor: a frame that kept its range keeps the slide default and is never flattened. The ceiling: a frame that lost most of its range is stretched at most 1.5 times, so grain and stain do not grow into the picture |
| `ADAPTIVE_MIDTONE` | 1.0 D | where the median prints: about a normally exposed slide's mid gray, and where it prints at the slide defaults (Print Density 1, ISO-R 115 print each density as itself) |

Contrast is `ADAPTIVE_SPAN / span` held to the range, then Grade = 115 / contrast on its 1-step grid, then Print Density puts the median at `ADAPTIVE_MIDTONE` given that Grade (step 0.01, held to 0 to 2). The fixtures' scene is dense (median 1.7 D), so Adaptive brightens it.

The Grade each option picks (fresh Estimate, Crossover 100 %, Strength 100 %; "uniform s" scales all three layers by s, a pure loss of range; dE is the 9-patch mean against the unfaded film):

| scanner | fade | span D | NegPy Density / Grade | Adaptive Density / Grade | dE Off / NC / NegPy / Adaptive |
|---|---|---|---|---|---|
| Coolscan-like | uniform 1 | 2.89 | 0.41 / 104 | 0.00 / 115 | 0.0 / 16.5 / 16.1 / 28.2 |
|  | uniform 0.8 | 2.32 | 0.58 / 89 | 0.41 / 111 | 13.1 / 23.4 / 23.5 / 30.9 |
|  | uniform 0.6 | 1.74 | 0.98 / 63 | 0.86 / 83 | 29.4 / 23.9 / 23.9 / 31.3 |
|  | uniform 0.45 | 1.31 | 1.12 / 50 | 1.26 / 77 | 42.9 / 35.1 / 35.0 / 30.4 |
|  | uniform 0.3 | 0.87 | 1.49 / 50 | 1.69 / 77 | 55.8 / 41.8 / 36.9 / 30.1 |
|  | linear 1/.7/.5 | 2.89 | 0.41 / 104 | 0.00 / 115 | 0.3 / 16.5 / 16.2 / 28.2 |
|  | crossover | 2.90 | 0.43 / 105 | 0.00 / 115 | 5.4 / 19.0 / 19.0 / 30.9 |
|  | low range .6/.45/.35 | 1.74 | 0.98 / 63 | 0.87 / 83 | 29.5 / 24.0 / 23.9 / 30.9 |
|  | low range .4/.3/.25 | 1.16 | 1.23 / 50 | 1.41 / 77 | 47.5 / 38.4 / 36.6 / 29.9 |
| Plustek-like | uniform 1 | 2.89 | 0.00 / 89 | 0.00 / 115 | 0.0 / 30.6 / 16.1 / 17.1 |
|  | uniform 0.8 | 2.32 | 0.00 / 90 | 0.00 / 111 | 5.6 / 33.0 / 26.7 / 26.4 |
|  | uniform 0.6 | 1.74 | 0.19 / 73 | 0.21 / 83 | 13.3 / 36.3 / 35.7 / 33.8 |
|  | uniform 0.45 | 1.31 | 0.54 / 54 | 0.61 / 77 | 20.8 / 36.3 / 35.9 / 30.4 |
|  | uniform 0.3 | 0.87 | 0.85 / 50 | 1.04 / 77 | 30.2 / 36.5 / 36.5 / 26.7 |
|  | linear 1/.7/.5 | 2.56 | 0.00 / 81 | 0.00 / 115 | 1.5 / 30.8 / 16.8 / 18.2 |
|  | crossover | 2.77 | 0.00 / 86 | 0.00 / 115 | 1.6 / 31.9 / 18.4 / 19.0 |
|  | low range .6/.45/.35 | 1.55 | 0.24 / 65 | 0.27 / 77 | 14.2 / 35.8 / 35.4 / 32.7 |
|  | low range .4/.3/.25 | 1.02 | 0.68 / 50 | 0.81 / 77 | 24.6 / 35.9 / 34.9 / 27.7 |

- **Adaptive never goes below ISO-R 77; NegPy reaches the ISO-R 50 floor at uniform 0.45 and below** (and on low range .4/.3/.25), as on the real files.
- On a heavily faded frame Adaptive lands closest to the unfaded film (Coolscan uniform 0.3: dE 30.1 against NegPy's 36.9 and NC's 41.8). On a mild fade NegPy's fit is closer, because Adaptive moves a frame's brightness to its midtone rule.
- On the Plustek-like scan both reach the Print Density floor (0) on any frame that kept its range: the dim white puts the whole frame too dense for the slider. The scanner zero (phase 3) is the control for that.

### 4. Fog diagnostics

`fade_fog.fit_fog` returns the measurement with what it read (`FogFit`): the grid cells, their shared floor, which were coherent and which were kept as clear film, the surface's range over its region before the cap, and why no surface was taken. `measure_fog` and the bake use it unchanged. Measure-only metrics: each render with fog NegPy reports `metrics["fade_fog"] = {amplitude, cap_hit, kept_share, reason}` (CPU; the bake runs before both engines). `compare.py` writes the overlay (`.p4-fog.png`): the scan as loaded with the removed density as a magenta tint (full at 0.15 D), each grid cell as a dot (green kept, orange read as picture, red incoherent), the removed map alone, and the render. A canvas overlay was not added: the sheet needs no app change and runs on your files.

Scenes (`phase4.bright_fog_scene`, unfaded film, Coolscan-like, a textured picture filling the frame): a bright area (none, a lit near-white wall over the right third, a pale sky band over the top third) with no fog, gradual fog from the left (0.12 D), a corner bump (0.14 D) or strong fog from the left (0.25 D, past the cap). "err" is the RMS of the removed map against the true uneven fog; "bright" the removed density over the bright area minus over the rest, with the truth in brackets.

| bright area | fog | amp before cap | cap hit | kept | err D | bright D |
|---|---|---|---|---|---|---|
| none | none | 0.003 | | 100 % | 0.000 | |
| none | left | 0.113 | | 100 % | 0.006 | |
| none | strong | 0.238 | yes | 100 % | 0.008 | |
| wall | none | 0.026 | | 44 % | 0.020 | −0.021 (0) |
| wall | left | 0.067 | | 44 % | 0.022 | −0.056 (−0.060) |
| sky | none | **0.102** | | 52 % | **0.075** | **−0.081 (0)** |
| sky | left | **0.206** | **yes** | 50 % | 0.058 | −0.065 (0) |
| sky | corner | 0.177 | yes | 52 % | 0.090 | −0.079 (+0.025) |

**The surface follows the picture when a pale area sits beside textured content.** The textured picture's thinnest pixel in each erosion window is denser than clear film, and the pale sky's is not, so the floor steps at the sky's edge. With no fog at all, the bake removes 0.10 D from everything but the sky; with real fog it hits the cap. That fits the real files, which hit the cap often. A lit wall does much less (0.026 D).

Stricter rejection, the same scenes (mean err over the 9 fogged scenes / the 3 clean ones, mean bright-area error):

| variant | fog scenes | clean scenes | bright area | strong fog alone |
|---|---|---|---|---|
| default | 0.029 | 0.032 | 0.026 | removes 0.15 (capped), err 0.008 |
| `FOG_PLAUSIBLE` 0.20 | 0.050 | 0.023 | 0.022 | rejected, err 0.116 |
| `FOG_PLAUSIBLE` 0.14 | 0.047 | 0.016 | 0.018 | rejected, err 0.116 |
| clear-cell band 0.14, bend limit kept at 0.28 | 0.027 | 0.016 | 0.023 | removes 0.15, err 0.011 |
| **clear-cell band 0.10**, bend limit kept at 0.28 | **0.025** | **0.008** | **0.019** | removes 0.118, err 0.024 |
| kept-cell minimum 64 or 160 | 0.029 | 0.032 | 0.026 | unchanged |
| clear cells on both sides of the gradient | 0.029 | 0.032 | 0.026 | unchanged |

- `FOG_PLAUSIBLE` does two jobs: the bend limit and the clear-cell band. Tightening it rejects a strong real fog outright. Splitting them (`fit_fog(clear_band=...)`) keeps strong fog and tightens only which cells count as clear film.
- A clear-cell band of 0.10 D cuts the clean-sky false removal from 0.10 to 0.03 D and the fogged-scene error a little; the cost is that a strong fog with little clear film is removed less (0.118 instead of 0.15).
- The kept-cell minimum and the both-sides rule change nothing here: a textured frame has clear-looking cells everywhere.
- Defaults are unchanged and the 0.15 D cap stays. The candidate is a clear band of 0.10 D; judge it on real files with the `.p4-fog.png` overlay first.

### 5. compare.py

`compare.py` now writes the phase 4 sheets by default (`--phase 3` or `all` for the earlier ones), with a `.p4-` prefix: `.p4-mix.png` (Neutrals NegPy and No Sky at Mix 0, 50, 100 %, each with a fresh Estimate, Crossover 100 %), `.p4-tone.png` (Tone Off, then NegPy and Adaptive at Strength 50 and 100 %, labeled with the Print Density and Grade each writes), `.p4-fog.png` (the overlay above). It prints each Estimate and the fog fit's metrics. Checked on a synthetic 16-bit scanner TIFF only.

### Tests

`make all`: format, Rust checks (81 passed), lint and ty pass; pytest 13,496 passed, 56 skipped, 2 failed. The two failures are `test_gradient_icon.py::test_a_checked_glyph_is_flat` and `test_a_toggle_icon_is_gray_while_off_and_brightens_when_checked`, which fail the same way at the branch tip in this container. GPU parity (`tests/test_gpu_fade_crossover_parity.py`) passes with no skips on the container's software adapter (llvmpipe, Mesa); rerun it on Metal. New tests: `test_fade_neutrals.py`, and additions to `test_fade_tone.py`, `test_fade_fog.py` and `test_sensor_sidebar.py`.

### Open

- The cold lean on your real files may have a cause these fixtures lack. The sky sector is the one mechanism found; No Sky is the A/B for it.
- Adaptive's constants are set from what a normal slide holds, not fitted to files. `ADAPTIVE_MIDTONE` decides brightness on every frame.
- Fog: the clear-band candidate is untested on real files.
- Tone Strength under a roll push: like Tone, the push carries the choice and the strength, and writes no frame's sliders.


## Phase 3

Every new choice is a `SegmentedChoice` in the Compare fold, a `ProcessConfig` field on the `fade` roll card (in `LATE_FADE_DEFAULTS`), in the Fade Restoration paste row and the card reset, with a REGISTRY cycle action (no default key). At the defaults (`full_scale`, tone `off`, fog `off`) the render is bit-identical: `test_a_scanner_file_at_full_scale_renders_as_before`, `test_the_defaults_render_as_before`, and the fog bake returns its input object.

### 0. NegPy applies Crossover; Hybrid is retired

`crossover_strength` is now Crossover on any slide whose survivals act; the NC model has no survivals, so no table. Crossover 0 is the survivals alone. `fade_ab_model` is `negpy` | `nc`. `migrations.RETIRED_FADE_MODELS = {"hybrid": "negpy"}` runs in `ProcessConfig.__post_init__` (edits, sticky settings, asset dicts) and in `rolls.roll_defaults` (a roll's stored default), as the crosstalk-profile renames do, because roll defaults overlay the field raw (`test_a_retired_hybrid_model_reads_as_negpy`, `test_a_roll_s_retired_hybrid_model_reads_as_negpy`). The phase 2 tables below now read "NegPy, Crossover 100 %" where they said Hybrid; `evaluate.py` reproduces them unchanged.

### 1. Scanner zero (`fade_ab_scanner_zero`)

Design: a clear zero on a scanner file tags its white space `scanner-<zero>-<unmix hash>` (`render_white_space`). Everything a camera RAW does with its white then follows with no new path: a render with no white for that tag measures one (`resolve_fade_white`, both engines, the tiled export's reference pass), Estimate reads density above the same anchor, the crossover tag changes, and paste and roll push drop the white (`_without_decode_white`, `DECODE_WHITE_FIELDS`). `density_anchor` reads the zero from the tag: `clear_film` is each channel's white, `clear_neutral` its mean log on all three. Remove Stain still sets its own per-channel anchor. The control hides for a camera RAW (`fade_white_space(cam_xyz) != ""`).

Fixtures (`tests/fade_crossover_scene.py`): **Plustek-like** = clear film 1.3 stops under full scale, channel densities mixed by `BROADBAND_MIX` (each channel reads 18–22 % of its neighbors); **Coolscan-like** = full scale, no mix. Fresh Estimate (Midtone) for every row, positive path. "Lean slope" is the share-weighted least-squares slope of each channel's band lean against mean log (`crossover_lean_fit`), "residual" the RMS after that line.

| scanner, fade | zero | survivals r/g/b | lean slope r/g/b | residual D | x | ramp C* max / mean | patch dE mean / max |
|---|---|---|---|---|---|---|---|
| Plustek, linear 1/.7/.5 | full_scale | 1.00/0.81/0.66 | +0.063/+0.003/−0.066 | 0.002 | 0 | 9.5 / 2.5 | 2.5 / 6.5 |
| | | | | | 1 | **1.3 / 0.2** | **1.5 / 2.8** |
| | clear_film | 1.00/0.77/0.58 | **+0.004/0.000/−0.004** | 0.002 | 0 = 1 | 6.3 / 2.4 | 3.8 / 6.3 |
| | clear_neutral | 1.00/0.73/0.53 | −0.044/−0.005/+0.050 | 0.001 | 1 | 6.1 / 0.7 | 2.4 / 4.2 |
| Plustek, crossover | full_scale | 1.00/0.94/0.79 | −0.042/−0.064/+0.106 | 0.034 | 1 | 7.0 / 0.9 | **1.6 / 3.2** |
| | clear_film | 1.00/0.92/0.78 | −0.056/−0.063/+0.118 | 0.030 | 1 | 7.0 / 2.5 | 3.5 / 8.0 |
| | clear_neutral | 1.00/0.91/0.73 | −0.080/−0.078/+0.159 | 0.036 | 1 | 13.6 / 1.5 | 2.4 / 4.4 |
| Coolscan, linear | full_scale | 1.00/0.70/0.50 | +0.003/+0.001/−0.004 | 0.000 | 0 | **0.3 / 0.2** | **0.2 / 0.3** |
| | clear_film | 1.00/0.70/0.50 | +0.004/+0.001/−0.005 | 0.002 | 0 | 10.7 / 4.0 | 4.9 / 8.8 |
| | clear_neutral | 1.00/0.67/0.44 | −0.054/−0.006/+0.060 | 0.001 | 1 | 10.8 / 1.5 | 2.6 / 6.4 |
| Coolscan, crossover | full_scale | 1.00/0.95/0.69 | −0.055/−0.140/+0.196 | 0.059 | 1 | 17.0 / 2.2 | **4.0 / 6.8** |
| | clear_film | 1.00/0.93/0.73 | −0.049/−0.125/+0.174 | 0.053 | 1 | 16.5 / 5.7 | 7.3 / 17.0 |
| | clear_neutral | 1.00/0.95/0.65 | −0.071/−0.146/+0.217 | 0.068 | 1 | 25.2 / 3.5 | 4.9 / 8.0 |

What it says:
- **The hypothesis holds for the lean**: on the dim broadband white, full scale turns a linear fade into a linear lean (slope ±0.065, residual 0.002 D); about the clear film the lean is gone (slope ±0.004), and the true ratios (0.7/0.5) come closer (0.77/0.58).
- **But the render is worse about the measured white.** The white a slide shows is its brightest picture, d_min of the faded dyes, not clear film. Scaling about it leaves the faded d_min's own color (a constant cast, C* 6–11), which neither the survivals nor the table remove (the lean is under `CROSSOVER_FLOOR`). Full scale plus Crossover 100 % absorbs the linear lean in the table and wins on every fixture.
- `clear_neutral` removes the per-channel part (no d_min cast), but its anchor sits off the true zero by the white's mean, so it leaves a smaller lean of the opposite sign. It is second best on Plustek and worst on Coolscan.
- On Coolscan-like scans full scale is the true zero, and either clear zero only adds error, as expected.

Real files (`compare.py`, `<name>.p3-zero.png`; scanners from each TIFF's Make, Model and Software tags):

| file | scanner | lean slope r/g/b: full scale | clear film | gray | by eye |
|---|---|---|---|---|---|
| NORAD 106-04 | Plustek OpticFilm 7600i, VueScan 9 | +0.115/+0.092/−0.207 (0.52 D) | **+0.014/+0.022/−0.036** (0.27 D) | +0.092/+0.100/−0.193 (0.39 D) | full scale and gray at Crossover 100 % most neutral; clear film keeps the stained white's yellow |
| FADED_SLIDES Image (2), (5), (9), (13) | Plustek OpticFilm 8200i, SilverFast 8.8 | ≤ ±0.06 | ≤ ±0.08 | ≤ ±0.09 | no zero removes a lean; the three look alike after Crossover |
| toppenhaug 19, 23, 30, 40 | Nikon Super Coolscan 4000 ED, Nikon Scan 4.0.3 | ≤ ±0.13 | about the same | about the same | full scale is the right zero, as expected for a full-scale white |
| toppenhaug 50 | same | −0.001/−0.147/+0.148 | **−0.011/+0.019/−0.007** | −0.132/−0.239/+0.372 (blue at the 0.2 bound) | **clear film is the most neutral at Crossover 0**: sand and sky lose the green-yellow lean that full scale leaves |

- The hypothesis (a dim white turns the fade into a linear lean) holds on NORAD, a VueScan file, not a SilverFast HDR file. The four SilverFast files show no such lean, so their whites sit near full scale or the effect is small there.
- Toppenhaug 50 is the counter-example to the fixtures: on this Coolscan frame the per-channel anchor removes a large lean and looks best. That frame is a heavily faded overcast beach whose brightest area is cloud; the white reads close to clear film there.
- The scans in `Faded slides/` other than NORAD are edited web positives and were not used.

### 2. Tone NegPy (`fade_ab_tone = negpy`)

Design: no hidden curve. The render reports `fade_tone_points`, the 0.5, 5, 10 … 90, 95 and 99.5 % points of the per-cell mean of the meter grid after the fade and the table, as capture density through the slide window with its White/Black Point (both engines; a GPU test checks they agree within 0.01 D). Choosing NegPy writes **Print Density** and **ISO-R Grade** (`fade_tone.tone_values`) and turns Auto Density and Auto Grade off. The controls a slide's tone has are Print Density (exposure, 2 stops per unit), ISO-R Grade (contrast about the 0.75 D pivot), Toe/Shoulder and White/Black Point. White/Black Point are ±0.25 D and on the roll's Process card, so a per-frame tone there would lock that card; Toe/Shoulder are roll-offs, not levels. Print Density and Grade are the Tone card's own, per frame, and are what Auto Density and Grade drive.

Translation of their units: their tone is levels at 0.5/99.5 %, a display gamma that takes the leveled median toward 0.42 (brightness −100·log2 γ, held to −25..60), and a tanh S-curve at strength 20 + (1 − range)·30 % (range = their display span lum_high − lum_low, held to 10..45). `tone_values` evaluates exactly that curve (`their_tone`) on the display values our curve shows at the slide defaults, then searches every Print Density (step 0.01, 0..2) and ISO-R Grade (step 1, 50..180) for the pair whose transfer curve shows the 13 points closest in display, least squares. So the contrast that grows with the missing range is their own rule, and it reaches us as a lower Grade; the 0.42 midtone is their display target on our curve's display, including a camera RAW slide's baseline and display rendering. A first, direct translation (levels onto the 3.0 D window, the S-curve's midtone slope 2/tanh 2 on Grade) gave ISO-R 50–58 on NORAD, far harder than their own tone; the fit gives 81.

Exact restore: `fade_tone_saved` holds (density, grade, auto_exposure, auto_normalize_contrast) as they were when NegPy was chosen. Leaving NegPy (to Off or NC) writes them back in the same edit (`AppController.set_fade_tone`) and clears the field, so Off restores exactly what the controls held (`test_leaving_negpy_restores_the_sliders_exactly`, `test_tone_negpy_writes_the_sliders_and_off_gives_them_back_exactly`). An edit to the sliders while NegPy is on is overwritten on the way back; that is what "exactly what the controls held before" means. `fade_tone_saved` is per frame: not pasted, not on the roll card; the card reset clears it.

Fixtures, camera RAW slide path, distance from the unfaded film (patch dE76 over the 9 patches, ramp dL* mean):

| fade | color model | tone Off | tone NC | tone NegPy (Density / Grade) |
|---|---|---|---|---|
| crossover | NegPy, Crossover 0 | 13.2 / 0.7 | 21.3 / 9.3 | 19.6 / 8.0 (0.43 / 105) |
| crossover | NegPy, Crossover 100 % | **5.4 / 1.7** | 19.0 / 10.6 | 19.0 / 8.7 (0.43 / 105) |
| crossover | NC | 10.0 / 1.8 | 24.8 / 10.9 | 22.1 / 8.3 (0.49 / 97) |
| linear 1/.7/.5 | NegPy (0 = 100 %) | **0.3 / 0.0** | 16.5 / 9.9 | 16.2 / 7.5 (0.41 / 104) |
| linear 1/.7/.5 | NC | 20.8 / 9.1 | 24.2 / 11.0 | 26.1 / 11.3 (0.71 / 79) |
| low range .6/.45/.35 | NegPy (0 = 100 %) | 29.5 / 16.8 | 24.0 / 10.9 | **23.9 / 11.0** (0.98 / 63) |
| low range .6/.45/.35 | NC | 41.4 / 28.2 | 32.1 / 14.4 | 34.5 / 16.0 (1.10 / 52) |

- Like NC's, Tone NegPy moves every frame that kept its range away from the unfaded film (+15 to +19 dE). It is a look, not a restoration, and it lands where theirs does (within 3 dE), slightly closer in the ramp.
- On a low-range fade, where the Estimate leaves the lost range (red survival set so the best layer is 1.0), both tones move toward the unfaded film (29.5 → 24); this is the case the "more pleasing on faded, low-range files" finding is about.
- Real files (`<name>.p3-tone.png`): NORAD gets 0.78 / 81 under NegPy and looks close to tone NC, a little lighter. Toppenhaug 30 (low range, green) gets 1.57 / 50, the Grade floor, and its remaining green cast grows with the contrast; under the NC color table the same tone looks fine. Toppenhaug 19: 0.62 / 116, 23: 0.55 / 100, 40: 0.88 / 87, 50: 1.41 / 50. The SilverFast frames all land at ISO-R 50 to 73: their tone takes the Grade to its floor on most faded files, so the slider range, not their rule, limits the contrast there.

### 3. Uneven fog (`fade_ab_fog`)

`nc`: `nc_rescue.fog_stage` / `apply_spatial`, run before their color table on the display positive, with their analysis measured on the flattened samples (`analyzeExpiredFilm`'s `options.spatial`). OpenCV.js is replaced by `negpy_core` (`ellipse_kernel`, `morphology_ex` erode with the default constant border, `gaussian_blur` sigma 6.4 with a replicated border, `resize_area`). The normal equations are summed in their order and solved by their Gauss-Jordan loop, so the coefficients come out bit-equal before `round4`. Port check (`port_check.py <checkout> <node_modules>`, opencv-js 5.0.0-release.1, their commit ff304397): on their own fog scene and on a clean Coolscan-like scene, the maps agree within 2.4e-7, the rounded coefficients exactly, and every probe within 1.5e-5 levels (`tests/fixtures/nc_fog_port.json`, `test_the_fog_*`). Rerunning the phase 2 port check reproduced its fixture byte for byte.

`negpy`: `fade_fog.bake_fog`, a source bake after Flat Field (and the sensor unmix), before the defect repairs, in `run_pipeline`: one single-slot cache keyed on the precorrect key, token `|fog-negpy` in `source_hash`, no shader. Estimate runs `run_pipeline`, so it reads the baked frame. Constants and why:

| constant | value | why |
|---|---|---|
| small copy, grid | 160 px wide, 32-wide grid, 2 % border | theirs |
| erosion, blur | ellipse 22 % of the width, sigma 4 % | theirs; erosion on density (`-log10` of the area-averaged linear light, pixels past the film left out) |
| fit scales above / below | 0.013 / 0.044 D | their 0.012 / 0.04 display units at the floor, where a slide's thinnest film shows about 0.9 display units per D |
| plausible bend | 0.28 D | their 0.25 the same way |
| clear-film cells | densest channel floor within 0.28 D of the grid's 5th percentile | a colored area is thin in one channel only; without this a blue sky over a gray wall read as 0.15 D of fog |
| coherence | floor of the mean within 0.02 D of the mean of the channel floors | on clear film one pixel is the thinnest in all channels; a window over two areas takes its channel floors from different pixels. Without it the sky's edge read as 0.045 D |
| region | the bounding box of the clear cells | the surface is trusted only where it was measured, as theirs is held at the region's edges |
| cap | 0.15 D | half a stop: a larger step across a slide is more often the picture, and it keeps a wrong fit a mild error, as their 0.15 limit does |
| gate | 0.02 D | under the grain of a scan and under 1 dE on a mid gray; a clean scan stays bit-identical |
| color | per-channel robust plane, zero-sum, × 0.5, held to 0.03 D over the region, zero at its centre | content in one channel can move color by at most 0.03 D |

Fixtures (`fog_scene`, Coolscan-like positive; "gradient left" is the spread of the mean dL* against the clean scan over eight column bands of the gray cells, "C* added" the largest chroma added on a 16-px gray block):

| scene | fog | gradient left (dL*) | wall dL* | C* added on gray (hue) |
|---|---|---|---|---|
| fog from the left edge + top-right corner, light wall | off | 1.62 | −0.83 | 0.0 |
| | nc | 1.65 | −1.44 | 0.0 |
| | negpy | **0.29** | −0.53 | 0.0 |
| clean: light wall, dark doorway, large warm area | nc | 0.44 (whole frame −14 L*) | −9.2 | **15.8 (−74°, blue)** |
| | negpy | 0 (bit-identical) | 0 | 0 |
| clean: light wall, dark doorway, blue sky | nc | 0.82 (−12 L*) | −9.3 | **23.4 (−56°, blue)** |
| | negpy | 0 (bit-identical) | 0 | 0 |
| their fog: blacks lifted in the positive | off | 5.77 | 0.3 | 0.0 |
| | nc | **0.10** | −0.3 | 0.0 |
| | negpy | 5.38 | 2.4 | 0.0 |

- **The blue spots reproduce.** Their scene (a textured full-range ramp) does not: every erosion window holds a black, so the floor is even. A real frame has flat areas. A light gray wall with a small dark doorway, under a large warm area or a blue sky, makes their per-channel floor read high in one or two channels over part of the frame, and the wall turns blue (15.8 and 23.4 C*) and 9–14 L* darker. NegPy's bake leaves both scans bit-identical.
- **The two stages remove different fogs.** Slide fog is added density, which shows in the clear film (the highlights); NegPy's bake removes it (1.62 → 0.29) and NC's, which erodes the positive's dark floor, does not touch it. NC removes its own model (a lifted floor in the positive, the fog of a negative) perfectly, and NegPy's does not, since that fog never reaches the thinnest film.
- Real files (`<name>.p3-fog.png`): NegPy's bake reads 0.08 to 0.15 D on most frames (the cap on toppenhaug 19 and three SilverFast frames; none on toppenhaug 50). By eye it lightens a band a little; whether that is fog or a lit area is not settled, and the cap is reached often enough that the surface may follow content. NC's stage darkens the frame and its edges (toppenhaug 19, SilverFast Image (9)). On these five Coolscan frames NC did not paint visible blue spots; the reported spots came from other frames.

### 4. compare.py

`compare.py` writes beside each input: `.crossover-0/0.5/1.png` (strength strips), `.model.png` (NegPy at Crossover 0 and 100 %, NC), `.zero.png` (three zeros × Crossover 0 and 100 %, fresh Estimate each, scanner files only), `.tone.png` (three tones × color models NegPy 100 % and NC), `.fog.png` (three fogs, Estimate read after the bake). It prints each Estimate's band table and lean line. The outputs carry a `.p3-` prefix so they never replace another run's files. Written for NORAD/106-04.tif, toppenhaug 19, 23, 30, 40 and 50 (Coolscan) and FADED_SLIDES Image (2), (5), (9) and (13) (SilverFast).

### Decisions to check

- Scanner zero: the clear zeros scale about the measured white, which on a slide is the faded d_min, not clear film. That is why they lose on every fixture despite removing the lean. A zero read from the mount gap or unexposed leader (true clear film) would test the hypothesis cleanly; the scans here have none.
- Tone NegPy is fitted to their curve, so it inherits their look. It writes only Print Density and Grade; their S-curve's roll-off at the ends has no counterpart there (Toe/Shoulder would be the place).
- A roll push of Tone NegPy carries the choice but writes no frame's sliders: `fade_tone_saved` is per frame. Choose NegPy again on that frame to write them.
- The fog bake uses the whole source frame (before geometry and crop), with a 2 % border, as their sample uses the analysis region.

### Not tested

- Coolscan blue spots on a real frame: not reproduced on the five toppenhaug frames.
- A real file with clear film (a mount gap or leader) to read the true zero from.
- The app driven by hand: the panel is covered by headless sidebar tests only.
- GPU: parity ran on this Mac's Metal adapter with no skips (`tests/test_gpu_fade_crossover_parity.py`, 9 tests: stored and stale table, tone points, each scanner zero at 0 and 100 %, the stale scanner white, the fog bake, the tiled export). The NC stages remain CPU-only.


## What changed in phase 2

| area | change |
|---|---|
| Cache keys | `fade_crossover`, `fade_crossover_table`, `fade_crossover_space` and `fade_ab_model` are in the CPU base-stage key (`engine.py`) and the GPU analysis-cache key. `GPUEngine`'s stage diff already re-runs from the exposure pass on any `process` change. |
| Paste and reset | `fade_crossover` and the three `fade_ab_*` fields are in the Fade Restoration paste row (`settings_catalog.py`), the card reset (`_SENSOR_FIELDS`) and the `fade` roll card. The table and its tag are in none of them: they are this frame's reading. |
| Meter area | `crossover_space(density, white_space, process)` hashes `analysis_buffer`, `analysis_rect` and `meter_rect`, so a new meter area measures the table again. |
| Table zero | The table is zero at the Estimate's midtone (the survivals' own fit point), not at the share-weighted mean lean. The stored table starts with that mean log; a render that measures for itself takes its own neutral-axis midtone. |
| Gates | No row acts under `CROSSOVER_FLOOR` = 0.04 D (the source's 0.02 display threshold for naming a cast, at a mid gray) or when the survivals do not act (`crossover_strength`). The table acts under color model **NegPy** (phase 2: Hybrid). |
| UI | **Crossover** slider (0–100 %) under Estimate, grayed out until the frame has a table and the model is NegPy (phase 2: Hybrid). One hint line names the crossover in D and the shadow and highlight hues (a port of `castName`). A closed **Compare** fold holds three `SegmentedChoice` controls. All four controls are on the `fade` roll card, with three REGISTRY cycle actions (no default key), guide bullets and all four catalogs. |
| A/B | `fade_ab_neutrals` (`negpy`/`nc`), `fade_ab_model` (`negpy`/`nc`; phase 2 also had `hybrid`, retired in phase 3), `fade_ab_tone` (`off`/`nc`). The port is in `negpy/features/process/nc_rescue.py`. |

## Bug fixes, before and after

| bug | before | after |
|---|---|---|
| Base-stage cache key: one engine, Crossover 0 then 1 | max diff 0.0 (CPU, reported on the Mac) | CPU 0.120, GPU 0.130 (`test_one_engine_follows_a_strength_change`, `test_one_gpu_engine_follows_a_strength_change`) |
| Paste and reset | strength not pasted, not reset | strength and the A/B fields pasted, reset and pushed by Roll; table never (`test_paste_carries_the_strength_and_never_the_table`, `test_roll_pushes_the_strength_and_never_the_table`) |
| Meter area | a new meter area kept the old table | the tag changes, so the table is measured again (`test_a_new_meter_area_measures_the_table_again`) |
| Rolls made before these fields (found by `make all`) | the next Fade edit on such a roll locked the card, since the roll never set `fade_crossover` or `fade_ab_*` | a roll that holds Fade defaults reads the new fields at their defaults (`rolls.LATE_FADE_DEFAULTS`); a roll that never set the card still counts a first edit as a divergence |

### Zero at the midtone and the gates (results 2 and 4 of phase 1)

Crossover fade, ramp C\* at crossover 1 by band, darkest first (max / mean):

| | b0 | b1 | b2 | b3 | b4 | b5 | b6 | b7 | worst |
|---|---|---|---|---|---|---|---|---|---|
| crossover 0 | 0.8 / 0.7 | 1.1 / 1.0 | 2.2 / 1.7 | 5.1 / 4.1 | 29.4 / 14.3 | 35.4 / 32.1 | 20.7 / 14.6 | 6.9 / 4.7 | 35.4 |
| 1, before (cast zero) | 0.2 / 0.2 | 0.3 / 0.2 | 0.4 / 0.3 | 1.8 / 0.8 | 7.1 / 4.7 | 8.8 / 6.9 | 4.4 / 3.2 | 3.6 / 2.2 | 8.8 |
| 1, after (midtone zero) | 0.2 / 0.2 | 0.3 / 0.2 | 0.4 / 0.3 | 0.1 / 0.1 | 2.3 / 1.0 | 3.6 / 2.3 | 1.3 / 0.9 | 5.5 / 4.3 | 5.5 |

The midtones are now exact, because the table is zero where the survivals were fit. The brightest band is worse (3.6 → 5.5): it lies beyond the last band anchor, where the table holds flat.

**Result 2, large red subject** (about 45 % of the frame), crossover 1:

| | b4 | b5 | b6 | b7 | red subject ΔC\* vs x=0 | red subject Δh |
|---|---|---|---|---|---|---|
| before | 7.3 / 6.6 | 7.8 / 6.9 | 6.9 / 5.4 | 7.9 / 6.9 | −2.5 | 0° |
| after | 7.1 / 3.3 | 7.9 / 5.2 | 1.6 / 1.2 | 3.8 / 2.4 | −4.2 | −1° |

The subject no longer shifts the whole table through the shares. The highlights recover (b6 6.9 → 1.6, b7 7.9 → 3.8). The bands it fills (b4, b5) keep some error, and the subject itself loses 4.2 C\* (hue unchanged): it sits in the band whose zero moved.

**Result 4, two opposite groups on an unfaded film:**

| | ramp worst C\* at 1 | big group C\* (0 → 1) | small group C\* (0 → 1) |
|---|---|---|---|
| before | 9.8 | 17.0 → 7.2 | 22.8 → 30.6 |
| after | 0.0 | 17.0 → 17.0 | 22.8 → 22.8 |

Estimate finds no fade, so the survivals stay at 1 and the gate keeps the table off. The render is bit-identical at every strength (`test_a_frame_with_no_fade_takes_no_table`). The mean-shift still settles on the larger group (`test_opposite_groups_at_one_density_follow_the_larger`). This only matters on a faded frame that also holds such a pair.

The linear-fade case keeps every band lean ≤ 0.011 D, so its crossover is under the floor and no row acts (`test_a_crossover_under_the_floor_applies_nothing`).

## A/B against NegativeConverter

All variants at crossover 1, Midtone Estimate. Patch ΔE76 is against the unfaded film over the 9 sky, foliage and skin patches.

**Crossover fade:**

| variant | neutrals | ramp C\* max | ramp C\* mean | patch ΔE mean | patch ΔE max |
|---|---|---|---|---|---|
| NegPy, Crossover 0 | negpy | 35.4 | 9.1 | 13.2 | 34.6 |
| NegPy, Crossover 100 % | negpy | 5.5 | 1.2 | **5.4** | **8.5** |
| NC model | negpy | **5.4** | **1.1** | 10.0 | 23.8 |
| NegPy, Crossover 0 + NC tone | negpy | 32.1 | 9.7 | 21.3 | 30.7 |
| NegPy, Crossover 100 % + NC tone | negpy | 3.3 | 1.4 | 19.0 | 26.3 |
| NC model + NC tone | negpy | 4.6 | 1.4 | 24.8 | 36.6 |
| NegPy, Crossover 0 | nc | 40.4 | 10.3 | 15.5 | 37.7 |
| NegPy, Crossover 100 % | nc | 8.4 | 2.1 | 5.8 | 11.7 |
| NegPy, Crossover 0 + NC tone | nc | 38.2 | 10.3 | 22.5 | 35.6 |
| NegPy, Crossover 100 % + NC tone | nc | 7.4 | 2.0 | 19.6 | 26.8 |

**Linear fade (survivals exact):**

| variant | neutrals | ramp C\* max | ramp C\* mean | patch ΔE mean | patch ΔE max |
|---|---|---|---|---|---|
| NegPy, Crossover 0 or 100 % | negpy | 0.6 | 0.2 | 0.3 | 0.6 |
| NC model | negpy | 3.4 | 1.5 | 20.8 | 29.8 |
| NegPy, Crossover 0 or 100 % + NC tone | negpy | 0.7 | 0.2 | 16.5 | 20.9 |
| NC model + NC tone | negpy | 3.6 | 1.1 | 24.2 | 38.1 |
| NegPy, Crossover 0 or 100 % | nc | 1.1 | 0.3 | 0.6 | 1.1 |

What the numbers say:

- **Neutralizing the grays**: the NC color table and NegPy at Crossover 100 % tie on this film (ramp max 5.4 vs 5.5).
- **Keeping colors right**: NegPy at Crossover 100 % wins clearly (patch ΔE 5.4 vs 10.0). NC works on the display positive with the survivals off, so it fixes the grays' hue but not the lost dye density. Saturated colors stay off, and on the linear fade NC is far worse (20.8 vs 0.3).
- **NC tone** moves every variant away from the unfaded film (ΔE +8 to +16). It is a look, levels and S-curve, not a restoration, and it does not undo a crossover by itself (NegPy + NC tone keeps C\* 32).
- **NC neutrals** give slightly worse survivals here than NegPy's neutral axis (NegPy at Crossover 100 % ramp max 8.4 vs 5.5). Their band refs pull the shadow ref from bands 1–2, where this film's crossover is largest.
- `tests/fade_crossover_scene.py` is my film model, not real film. The Statue NEF and the faded-slide folders decide this; `compare.py` makes the contact sheets for that.

## Port check

`experiments/fade_crossover/port_check/port_check.py <NegativeConverter checkout>` runs their `expiredRescue.js` (commit `ff3043976b7cb72266d89e659faa65142c21b635`) under Node 22.23.4 on two deterministic 16-bit inputs (`tests/nc_rescue_fixtures.py`):

- **faded**: their own test scene, aged by their test's fog/top/gamma model without its crossover term;
- **crossover**: our crossover film as a display positive.

It stores their output in `tests/fixtures/nc_rescue_port.json`; `tests/test_nc_rescue.py` compares against it with no Node needed. Result: the analysis matches field for field (lumLow 0.3521 / 0.0532, lumHigh 0.7876 / 0.7749, every band mean and the lean identical to 1e-4). All 30 probe pixels match to within 0.0000 levels for colour only, tone only and both. Tolerances in the test: half a level for tables and curves, one level (1/255) for pixels.

What the port leaves out on purpose: the semantic-anchor weights (`options.anchors`, which need their ADE20K model) and the spatial stage (uneven fog, local contrast, which need OpenCV.js on their side). The Python stage reads a 16-bit quantization of our positive, as their 16-bit path does, and applies `applyExpiredTone` to floats.

How our engine feeds it: the stage runs after toning and the crop, before Finish. It encodes the scene-linear working image with the working TRC (Adobe RGB, 563/256), applies their stage in 0..255, and decodes again. Their app reads an sRGB-encoded positive, so the same pixel sits at a slightly different display value. That difference comes from the two pipelines, not from the port.

## GPU

- The crossover table renders in `normalization.wgsl`. Its uniform is a vec4 (lo, hi, on) plus 64 vec4 rows (normalization UBO 224 → 1264 bytes). The `nc` stages stay CPU-only: `ImageProcessor._cpu_only` routes a slide with `fade_ab_model = nc` or `fade_ab_tone = nc` to the CPU for preview, export and batch.
- Adapter here: **Mesa llvmpipe (LLVM 20.1.2, 256 bits), OpenGL 4.5 Core, Mesa 25.2.8**, which NegPy's `GPUDevice` reports as "CPU via OpenGL". Parity through `run_pipeline`: max 0.00001 at strengths 0, 0.5 and 1, identical tables when each engine measures its own, tiled export mean < 0.002. Rerun on Metal:

```bash
uv run pytest tests/test_gpu_fade_crossover_parity.py -v
```

## Density-unit constants

The source works on display-encoded 0..1 values. Near a mid gray (display ≈ 0.46), `dV/dD = ln10 · V / 2.2 ≈ 0.48` display units per D.

| constant | source | here | why |
|---|---|---|---|
| `CROSSOVER_WEIGHTS` | Rec. 709 luma | (⅓, ⅓, ⅓) | The table acts on dye densities. Equal weights keep each pixel's mean dye density; each row sums to zero under them (tested). |
| `CROSSOVER_NEUTRAL_SPREAD` | 0.12 | 0.25 D | 0.12 display ÷ 0.48. |
| `CROSSOVER_CLUSTER_RADIUS` | 0.06 | 0.12 D | 0.06 display ÷ 0.48. |
| `CROSSOVER_LIMIT` | 0.35 per channel | 0.3 D, whole row | The straight conversion is about 0.7 D, over a stop. Scaling the row keeps the zero sum and the hue. |
| `CROSSOVER_FLOOR` | 0.02 (cast naming) | 0.04 D | 0.02 display ÷ 0.48. |
| `CROSSOVER_MIN_SPAN` | 0.02 | 0.05 D | A frame needs some range to band. |
| band edges, 0.5/99.5 %, share 0.4 %, weight ≥ 8, 3 steps, 64 bins | | unchanged | Relative to the frame. |

## Storage and versions

- New `ProcessConfig` fields: `fade_crossover` (0), `fade_crossover_table` (None), `fade_crossover_space` (""), `fade_ab_neutrals` (`negpy`), `fade_ab_model` (`negpy`), `fade_ab_tone` (`off`). No migration: new keys with neutral defaults. `from_flat_dict` fills them; `__post_init__` restores the tuple and the enums.
- The stored table is `(zero, then 5 values per band)`. The phase 1 layout had no zero. Only this experiment branch ever wrote one, and a stale tag re-measures it.
- `THUMBNAIL_RENDER_VERSION` is not bumped: `claude/raw-density-order` already bumped it once over main (27 → 32), and the defaults render bit-identically (`test_the_defaults_render_as_before`, `test_crossover_at_zero_reads_nothing_and_changes_nothing`). Changed fields are hashed (`_changed_fields`), so a frame that uses any of them gets a new fingerprint.

## Open questions

- **Highlights beyond the last band.** The table holds flat past the outer anchors, and with the midtone zero that is now the largest leftover error (b7 5.5). Options: a short linear extrapolation of the last segment, or a band edge at 0.97.
- **Saturated subjects in a band.** The subject keeps its hue but loses about 4 C\* (result 2). Their neutral weight is soft (0.25 D at half weight). A harder weight would protect it, at the cost of fewer grays on a faded frame.
- **Cast Removal.** It meters the grid after the table, so it fits what the table leaves. Under NegPy at Crossover 100 % the neutral axis is near neutral and Cast Removal has little to do. A slide starts with Cast Removal at 0.
- **Automatic white balance.** A RAW's as-shot gains are inside the density affine and named in `render_white_space`, so a WB change re-measures the table. The NC stages see the rendered positive, so they follow every upstream change on each render.
- **Print (negatives).** Not run there. The table would belong after `normalize_log_image`, where a gray has equal values. The NC stages are gated to slides too (`nc_stage_flags`).
- **Crop and the NC stages.** The NC stages analyze the cropped positive, as their app does. The crop tool's full-frame preview shows the whole frame, so they read a slightly different image there.
- **Rust port.** `apply_crossover` is the only hot loop on a full image: a `par_zip` kernel over a `[[f32; 3]; 64]` table, with output pins. `neutral_bands` runs on the meter grid, and NumPy is fast enough there. The NC stages are an experiment and stay in NumPy.

## Not tested here

- Real files. This container has no RAW and no faded-slide scan; `compare.py` ran only on a synthetic 16-bit TIFF.
- A real GPU. Parity ran on llvmpipe only.
- The app driven end to end with a mouse. The sidebar tests build the panel headless; a screenshot confirmed the layout at panel width.

## Running it on real files

```bash
uv run python experiments/fade_crossover/compare.py ~/Pictures/Statue.NEF ~/Scans/faded/slide01.tif
```

The sheets are listed under Phase 3, section 4. Options: `--long-edge 2000`, `--method midtone|spread`, `--gpu` (the NC stages still render on the CPU).
