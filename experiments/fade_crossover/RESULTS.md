# Fade Restoration crossover table: prototype results

Branch `experiment/fade-crossover-table`, from `claude/raw-density-order`. CPU and GPU, no UI.

## What the prototype does

Fade Restoration scales each dye layer's density about the white (`F`). That fits a fade that is linear in density. It cannot fit a crossover, where the neutrals lean one way in the shadows and another way in the highlights. The prototype adds a residual table after `F`:

1. **Measure** (`fade.measure_crossover`): on the meter grid in log density **after** the fade matrix and the RAW anchor (`unmix_log_image(grid, stage.unmix, stage.anchor)`), before the primaries. 8 bands of the mean log `L̄ = (L_r + L_g + L_b)/3` over the frame's own 0.5–99.5 % range. Each band takes a neutral-weighted mean, then 3 mean-shift steps. A band's lean is its centre minus its `L̄`.
2. **Table** (`fade.crossover_offsets`): a Fritsch–Carlson cubic per channel through the band anchors, flat beyond the outer bands, sampled into 64 × 3 rows between the outer anchors. The offset is `-(cast·neutralize + (lean − cast)·crossover)`, with `neutralize = 0`.
3. **Apply** (`fade.apply_crossover`): each pixel adds the row that its own `L̄` selects, with linear interpolation between bins. This happens in `TransparencyBaseProcessor` after the fade matrix and before `relog_matrix_image(..., stage.post)`. The meters (Cast Removal, Auto Density and Grade) read the corrected grid.

Ported from NegativeConverter (`expiredRescue.js`: `analyzeExpiredFilm`, `offsetsFromBands`, `shapeCubic`), MIT, Copyright (c) 2025 lex. The notice is in `NOTICE.md`.

### Config and storage

| field | default | role |
|---|---|---|
| `fade_crossover` | `0.0` | strength 0..1; 0 reads and changes nothing |
| `fade_crossover_table` | `None` | flat tuple, 5 values per band: mean log, share, lean r, g, b |
| `fade_crossover_space` | `""` | `crossover_space(fade_density(...), white_space)`: hash of the fade matrix (affine incl. gains/white), the anchor and the white space |

- **Estimate** (`estimate_fade`) measures the table on the grid passed through its own survivals (`survival_grid`). The Estimate button stores the table with its tag. The table is per frame and locks no card, and it is not in `ROLL_DEFAULT_FIELDS`.
- **Render**: if `fade_crossover > 0` and the stored tag differs from the render's own, the render measures the table on its grid (`resolve_crossover`) and reports it as `metrics["fade_crossover"]`. The controller stores it (`_measured_fade_crossover`), the same pattern as the stale RAW white. So a slider change after Estimate does not lose the table: the next render measures it again against the new survivals.
- `fade_estimate_grid` forces `fade_crossover = 0`, so the Estimate grid never measures a table.
- Both engines render the table. The GPU measures a stale table on the same CPU meter grid as the RAW white, caches it per source, and reports it the same way. A tiled export takes the table from its reference pass.
- No migration is needed: these are new keys with neutral defaults, and `from_flat_dict` fills them. The tuple is restored in `ProcessConfig.__post_init__`. `test_the_table_survives_a_flat_round_trip` covers the round trip.
- **`THUMBNAIL_RENDER_VERSION` is not bumped.** CLAUDE.md allows one bump per branch over main, and `claude/raw-density-order` already bumped it (main 27 → 32). At the default `fade_crossover = 0`, the pixels are bit-identical (`test_crossover_at_zero_reads_nothing_and_changes_nothing`). The new fields are hashed only when changed (`_changed_fields`), so an edit that turns the table on gets a new fingerprint.
- `BOUNDS_VERSION` is unchanged: a slide's bounds are the fixed transfer window.

## Density-unit constants

The source works on display-encoded 0..1 values (sRGB-like). Near a mid gray (display ≈ 0.46), `dV/dD = ln10 · V / 2.2 ≈ 0.48` display units per D. I used that slope to convert the source constants, then adjusted where density needs it:

| constant | source | here | why |
|---|---|---|---|
| `CROSSOVER_WEIGHTS` | Rec. 709 luma | (⅓, ⅓, ⅓) | The table acts on dye densities, not display RGB. Equal weights keep each pixel's mean dye density, the neutral that the transfer curve renders. Each row sums to zero under them (tested). |
| `CROSSOVER_NEUTRAL_SPREAD` | 0.12 | **0.25 D** | 0.12 display ÷ 0.48 at a mid gray. |
| `CROSSOVER_CLUSTER_RADIUS` | 0.06 | **0.12 D** | 0.06 display ÷ 0.48. |
| `CROSSOVER_LIMIT` | 0.35 (per channel) | **0.3 D** (whole row) | The straight conversion gives about 0.7 D, which is more than a stop of color correction. After the survivals, a larger residual says the survivals are wrong, not that a crossover is present. The row is scaled as a whole, not clamped per channel: that keeps the zero sum and the hue of the correction. |
| `CROSSOVER_MIN_SPAN` | 0.02 (display) | 0.05 D | A frame with less than a twentieth of a D of range has nothing to band. |
| band edges, 0.5/99.5 %, share 0.4 %, weight ≥ 8, 3 steps, 64 bins | — | unchanged | These are relative to the frame, so they have no unit. |

A departure from the source: Fritsch–Carlson is nonlinear, so three per-channel cubics through zero-sum anchors do not sum to zero between the anchors. `crossover_offsets` takes the weighted mean out of each row after the cubic. The zero sum then holds exactly, and the bin index (`L̄`) does not change when the offset is applied.

## Test fixtures (not circular)

`tests/fade_crossover_scene.py` builds the scenes from an **independent** film model. Each dye layer has a logistic H&D curve with its own `d_max`, gamma and toe/shoulder position `e0`. The scene is a 32-step gray ramp, a second gray scale behind the patches, sky, foliage and skin at three exposures, and optional extras: a large red subject (about 45 % of the frame) or two opposite near-neutral groups. The crossover film:

| layer | d_max | gamma | e0 |
|---|---|---|---|
| R | 3.2 | 1.65 | −0.60 |
| G | 2.8 | 1.40 | −0.50 |
| B | 2.5 | 1.85 | −0.72 |

The reference film has the same layer in all three channels (3.3 / 1.7 / −0.6). Every render uses the real `DarkroomEngine` with the slide defaults (no meters, no Cast Removal). Estimate reads the same grid that the panel reads (`transfer_grid`), with the Midtone method (the panel default). C\* and hue come from the engine's own working-space Lab. Reproduce with:

```bash
uv run python experiments/fade_crossover/evaluate.py
```

## Results

### 1. Crossover fade, survivals only vs survivals + table

Estimate: red 1.0, green 0.954, blue 0.692. Table measured after those survivals (leans in D):

| band mean log | share | lean R | lean G | lean B |
|---|---|---|---|---|
| −3.094 | 0.027 | +0.094 | +0.318 | −0.413 |
| −2.878 | 0.020 | +0.113 | +0.293 | −0.406 |
| −2.491 | 0.020 | +0.097 | +0.193 | −0.289 |
| −2.043 | 0.078 | +0.032 | +0.066 | −0.098 |
| −1.695 | 0.097 | −0.015 | −0.025 | +0.040 |
| −1.264 | 0.063 | −0.051 | −0.110 | +0.161 |
| −0.732 | 0.080 | −0.067 | −0.140 | +0.207 |
| −0.417 | 0.152 | −0.039 | −0.093 | +0.132 |

Neutral ramp C\* (max / mean) in 8 groups of ramp steps, darkest first:

| crossover | b0 | b1 | b2 | b3 | b4 | b5 | b6 | b7 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.8 / 0.7 | 1.1 / 1.0 | 2.2 / 1.7 | 5.1 / 4.1 | 29.4 / 14.3 | 35.4 / 32.1 | 20.7 / 14.6 | 6.9 / 4.7 |
| 0.5 | 0.3 / 0.3 | 0.5 / 0.4 | 1.0 / 0.7 | 2.0 / 1.6 | 18.2 / 9.2 | 22.1 / 19.7 | 12.6 / 8.9 | 3.2 / 1.5 |
| 1 | 0.2 / 0.2 | 0.3 / 0.2 | 0.4 / 0.3 | 1.8 / 0.8 | 7.1 / 4.7 | 8.8 / 6.9 | 4.4 / 3.2 | 3.6 / 2.2 |

The worst ramp error falls from C\* 35.4 to 8.8 at full strength. The remaining error is in the upper midtones, where the shadow leans (up to 0.41 D) hit `CROSSOVER_LIMIT`. Over the brightest ramp steps, `L̄` is above the last band anchor (−0.417), so the table holds flat there. That is visible as a slight yellow-green in the brightest steps of `compare.py`'s output on the same film.

Colored patches, C\* / hue (°). Drift is at crossover 1 against crossover 0 and against the unfaded film:

| patch | unfaded | x=0 | x=0.5 | x=1 | ΔC\* vs 0 | Δh vs 0 | ΔC\* vs unfaded | Δh vs unfaded |
|---|---|---|---|---|---|---|---|---|
| sky −0.6 | 10.9 / −82 | 10.8 / −91 | 15.8 / −79 | 21.5 / −72 | +10.6 | +20 | +10.6 | +11 |
| sky 0 | 31.6 / −92 | 61.2 / −69 | 50.4 / −75 | 39.5 / −85 | −21.8 | −16 | +7.9 | +7 |
| sky +0.45 | 15.7 / −115 | 25.5 / −81 | 18.9 / −93 | 13.5 / −116 | −11.9 | −35 | −2.2 | −1 |
| foliage −0.6 | 1.3 / 144 | 3.2 / 142 | 1.7 / 140 | 1.3 / 140 | −1.8 | −2 | +0.1 | −4 |
| foliage 0 | 30.6 / 146 | 33.3 / 142 | 28.4 / 143 | 23.3 / 143 | −10.0 | +1 | −7.2 | −2 |
| foliage +0.45 | 70.1 / 137 | 60.0 / 135 | 65.9 / 134 | 71.4 / 134 | +11.3 | −1 | +1.3 | −3 |
| skin −0.6 | 4.3 / 36 | 6.1 / 70 | 4.9 / 57 | 3.9 / 39 | −2.2 | −31 | −0.4 | +3 |
| skin 0 | 27.6 / 49 | 25.8 / 9 | 25.0 / 30 | 27.2 / 49 | +1.5 | +40 | −0.4 | −1 |
| skin +0.45 | 15.1 / 61 | 24.0 / −30 | 15.2 / −14 | 10.3 / 30 | −13.7 | +61 | −4.8 | −31 |

The colors move a lot against crossover 0, but mostly **toward** the unfaded film. Skin at 0 goes from 40° hue error to 1°, and sky at +0.45 from 34° to 1°. The colors carry the same density-dependent crossover as the grays, so the table corrects them too. Two patches overshoot: dark sky (+11° / +10.6 C\*) and bright skin (−31°).

### 2. Large saturated subject

A red subject over about 45 % of the frame, which fills the bands around `L̄ ≈ −1.7`:

- Band leans move by at most 0.03 D (largest: band 5, R −0.051 → −0.017). The subject's spread is about 1 D, which gives it a neutral weight near 0.06, and the mean-shift keeps each band on the grays.
- The red subject itself: C\* 68.4 → 65.9 and hue 40° → 39° at crossover 1. It is not neutralized.
- Side effect: ramp C\* at crossover 1 is 7.3 / 7.8 / 6.9 / 7.9 for b4–b7 (3.6 at b7 without the subject). The subject adds share to the dense bands. The cast is the **share-weighted** mean lean, so it moves, and every band's residual (`lean − cast`) moves with it. See the open questions.

### 3. No crossover (linear fade 1.0 / 0.7 / 0.5)

Estimate reads 0.701 / 0.503. Every band lean is ≤ 0.011 D. Ramp C\* stays ≤ 1.0 at crossover 1 (≤ 0.6 at 0). Patches move by ≤ 0.4 C\* and ≤ 1°. The table is close to the identity, so it does not fight the survivals. The tests assert this (`test_a_linear_fade_leaves_the_table_near_identity`, `test_the_cast_term_is_left_to_the_survivals`).

### 4. Two groups leaning opposite ways at one density

The unfaded film, with two near-neutral groups at one exposure: a large magenta one (2/3 of the area) and a small green one. Estimate finds nothing to estimate, so the render measures the table itself. **The larger group wins:**

| | C\* at 0 | at 0.5 | at 1 |
|---|---|---|---|
| big group (magenta) | 17.0 | 12.3 | 7.2 |
| small group (green) | 22.8 | 26.6 | 30.6 |
| ramp worst (b4) | 0.0 | 4.8 | 9.8 |

The mean-shift settles on the larger cluster and the table pulls it to neutral. That pushes the smaller group and the true grays the other way. Both groups are within `CROSSOVER_NEUTRAL_SPREAD` (spread about 0.12 to 0.24 D), so a low-chroma subject such as a magenta-gray wall reads as a lean. This is the main failure mode for real photos.

## GPU

The table rides the normalization uniform (`NormUniforms`, 224 → 1264 bytes): a `crossover` vec4 (lo, hi, on, 0) and `crossover_rows: array<vec4<f32>, 64>`, packed by `fade.crossover_uniform`. A uniform is enough, so no texture is needed. `normalization.wgsl` adds the row after the unmix rows and the anchor, and before the post (primaries) rows, exactly where the CPU adds it. `L̄` is `(r + g + b) * (1/3)` in float32 on both engines. `apply_crossover` uses the same order, so the bin index agrees.

- Measurement: when the tag is stale, `GPUEngine.process_to_texture` builds the analysis source as it does for the white, takes `faded_stage_grid` (the grid after the density matrix and before the primaries) and calls `resolve_crossover`. The result is cached under the white's key plus the crossover space, and returned as `metrics["fade_crossover"]`.
- The slide meters (Cast Removal, Auto Density/Grade) read `density_stage_grid(..., crossover)`, the grid with the table applied, on both engines. The table is in the prefilter cache key.
- Tiled export: the reference pass measures the table. `_process_tiled` carries it onto the settings (`with_fade_crossover`), tiles never measure (`measure=False`), and the global meter grid applies it.
- The crossover fields are in the CPU base-stage key (`engine.py`) and the GPU analysis-cache key. The first commit missed them, so in a live session a Crossover change kept the cached base stage. `test_one_engine_follows_a_strength_change` covers it now.

Parity, measured here on Mesa llvmpipe (a software OpenGL device that wgpu accepts as "CPU via OpenGL"), on the 192 × 144 crossover scene through `ImageProcessor.run_pipeline`:

| case | max abs diff (display, 0..1) | mean |
|---|---|---|
| stored table, crossover 0 | 0.00000 | 0.000000 |
| stored table, crossover 0.5 | 0.00000 | 0.000000 |
| stored table, crossover 1 | 0.00001 | 0.000000 |
| stale table, each engine measures its own | 0.0000 (tables identical) | 0.00000 |

`tests/test_gpu_fade_crossover_parity.py` holds these cases and a tiled-vs-direct export, with looser tolerances (max 0.01, tiled mean 0.002) for real GPUs. llvmpipe computes in IEEE float32 on the CPU, so it matches closely. Metal on a Mac can differ in transcendental precision and FMA; run the test file there to confirm.

## Open questions

- **The cast reference.** `cast` is the share-weighted mean lean, so a subject that changes the band shares moves the whole table (result 2). Estimate fits the survivals on the midtone grays, so the lean at the Estimate's midtone density is probably the better zero: the table would then be exactly zero where the survivals were fit. A second option: weight the cast by band, without the shares.
- **Gating.** On a frame with no fade (result 4) the table only does harm. Options: apply it only when Estimate succeeded (`red is not None`), and skip it when the measured crossover (`max |lean − cast|`) is below a floor.
- **Cast Removal.** Cast Removal meters `prefiltered`, which is after the table, so it fits what the table leaves. With the table at 1 the neutral axis is close to neutral and Cast Removal has little to do. The two compose like Fade + Cast Removal, with no guard. They overlap in the job: Cast Removal is a per-channel gain/offset (linear in density) and the table is its nonlinear residual. A slide starts with Cast Removal at 0, so the default has no conflict.
- **Automatic white balance.** On a RAW, the as-shot WB gains are a log offset inside the density affine, before the table. The gains are part of `render_white_space`, so a WB change makes the tag stale and the render measures the table again. The filtration sliders (CMY) act later in `TransferProcessor`, so they neither move nor re-measure the table. The table does not track the analysis rectangle (the RAW white does not either): a new meter area keeps the old table until the matrix changes.
- **Print (negatives).** I would not run it at this point in the Print path. Before the bounds, a negative's neutrals are not neutral (the orange mask, per-channel bounds). On a negative the table would belong after `normalize_log_image`, measured on the normalized grid, where a gray has equal values. Expired C-41 has exactly this crossover, so this is worth a separate experiment. The cast term would then compete with Cast Removal and the per-channel bounds, which own the cast on a negative.
- **Highlights beyond the last band.** The table holds flat past the outer anchors. A bright, sparse highlight with its own lean keeps the last band's correction (visible in the synthetic `compare.py` output). Options: extrapolate the last segment linearly over a short distance, or add a band edge at 0.97.

## What a Rust port needs (CLAUDE.md: CPU kernels live in `negpy_core`)

- **`apply_crossover`** is the hot loop: an element-wise pass over the image. It needs a `par_zip` kernel in `src/fade.rs` (slices + a `[[f32; 3]; 64]` table + lo/hi), a wrapper in `src/py/fade.rs` behind `py::detach`, and output pins in `tests/test_core_fade.py`. Compute `L̄` in f32 in a fixed order (r + g + b, then × ⅓) so the GPU can match it. The NumPy prototype uses f64 there. Clamp the bin position before the integer cast (`trunc_bits`).
- **`measure_crossover`** runs on the meter grid (thousands of cells, not megapixels), so NumPy is probably fast enough. A port would need the linear percentile already in the core, weighted per-band sums in f64 with a fixed chunk order (one chunk per thread, merged in order), and the 3 mean-shift passes. It must keep NumPy's arithmetic, or the stored tables change: the band means feed a persisted value.
- The cubic and the 64-bin table are tiny and can stay in Python.
- `_mean_log` uses a NumPy matmul. On a pinned kernel that becomes an explicit sum, so BLAS never decides the order.

## Running it on real files

```bash
uv run python experiments/fade_crossover/compare.py ~/Pictures/Statue.NEF ~/Scans/faded/slide01.tif
```

The script writes `<name>.crossover-0.png`, `-0.5.png` and `-1.png` beside each input. It also prints the Estimate and the band table. Options: `--long-edge 2000` (downsample before the render), `--method midtone|spread`, `--gpu` (render on the GPU, as the preview does; the default is the CPU). Each file opens as a Slide the way the app opens it (a scanner TIFF starts Positive), with the slide defaults. A RAW gets its measured white, as the Estimate button stores it. I tested it here on a synthetic 16-bit TIFF only: this container has no RAW file.
