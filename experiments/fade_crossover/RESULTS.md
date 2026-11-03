# Fade Restoration crossover table and NegativeConverter A/B: results

Branch `experiment/fade-crossover-table`, from `claude/raw-density-order`. CPU and GPU for the table, the scanner zero and the fog bake; CPU only for the NegativeConverter stages. Defaults render bit-identically to the base branch.

Reproduce the numbers:

```bash
uv run python experiments/fade_crossover/evaluate.py      # phases 1 and 2
uv run python experiments/fade_crossover/phase3.py      # phase 3: scanner zero, tone, fog
```

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

Real files (`compare.py`, `<name>.zero.png`): **NORAD 106-04 (Plustek, SilverFast HDR raw)** confirms the lean: full scale reads slopes +0.115/+0.092/−0.207 and a 0.52 D crossover, clear film +0.014/+0.022/−0.036 and 0.27 D, gray +0.092/+0.100/−0.193 and 0.39 D. By eye, full scale and gray at Crossover 100 % give the most neutral picture; clear film keeps a yellow cast at both strengths, which is the stained white's own color. **Toppenhaug 19 and 30** show almost no lean under any zero (slopes ≤ 0.02 on 19; 30 has a real green/blue lean of ±0.13 that no zero moves), so those scans behave as full-scale scans.

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
- Real files (`<name>.tone.png`): NORAD gets 0.78 / 81 under NegPy and looks close to tone NC, a little lighter. Toppenhaug 30 (low range, green) gets 1.57 / 50, the Grade floor, and its remaining green cast grows with the contrast; under the NC color table the same tone looks fine. Toppenhaug 19: 0.62 / 116.

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
- Real files (`<name>.fog.png`): NORAD reads 0.084 D, toppenhaug 19 the 0.15 D cap, toppenhaug 30 0.14 D. By eye NegPy's bake changes these frames little; NC's darkens toppenhaug 19 overall. None of these files shows the blue spots, which were reported on Coolscan files.

### 4. compare.py

`compare.py` writes beside each input: `.crossover-0/0.5/1.png` (strength strips), `.model.png` (NegPy at Crossover 0 and 100 %, NC), `.zero.png` (three zeros × Crossover 0 and 100 %, fresh Estimate each, scanner files only), `.tone.png` (three tones × color models NegPy 100 % and NC), `.fog.png` (three fogs, Estimate read after the bake). It prints each Estimate's band table and lean line. Written for NORAD/106-04.tif, toppenhaug 19 and toppenhaug 30.

### Decisions to check

- Scanner zero: the clear zeros scale about the measured white, which on a slide is the faded d_min, not clear film. That is why they lose on every fixture despite removing the lean. A zero read from the mount gap or unexposed leader (true clear film) would test the hypothesis cleanly; the scans here have none.
- Tone NegPy is fitted to their curve, so it inherits their look. It writes only Print Density and Grade; their S-curve's roll-off at the ends has no counterpart there (Toe/Shoulder would be the place).
- A roll push of Tone NegPy carries the choice but writes no frame's sliders: `fade_tone_saved` is per frame. Choose NegPy again on that frame to write them.
- The fog bake uses the whole source frame (before geometry and crop), with a 2 % border, as their sample uses the analysis region.

### Not tested

- **Coolscan files**: I did not find them; the "Coolscan-like" findings are fixtures only, and the toppenhaug scanner is unknown to me.
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
