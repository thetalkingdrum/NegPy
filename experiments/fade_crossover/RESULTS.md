# Fade Restoration crossover table and NegativeConverter A/B: results

Branch `experiment/fade-crossover-table`, from `claude/raw-density-order`. CPU and GPU for the table; CPU only for the NegativeConverter stages. Defaults render bit-identically to the base branch.

Reproduce the numbers:

```bash
uv run python experiments/fade_crossover/evaluate.py
```

## What changed in phase 2

| area | change |
|---|---|
| Cache keys | `fade_crossover`, `fade_crossover_table`, `fade_crossover_space` and `fade_ab_model` are in the CPU base-stage key (`engine.py`) and the GPU analysis-cache key. `GPUEngine`'s stage diff already re-runs from the exposure pass on any `process` change. |
| Paste and reset | `fade_crossover` and the three `fade_ab_*` fields are in the Fade Restoration paste row (`settings_catalog.py`), the card reset (`_SENSOR_FIELDS`) and the `fade` roll card. The table and its tag are in none of them: they are this frame's reading. |
| Meter area | `crossover_space(density, white_space, process)` hashes `analysis_buffer`, `analysis_rect` and `meter_rect`, so a new meter area measures the table again. |
| Table zero | The table is zero at the Estimate's midtone (the survivals' own fit point), not at the share-weighted mean lean. The stored table starts with that mean log; a render that measures for itself takes its own neutral-axis midtone. |
| Gates | No row acts under `CROSSOVER_FLOOR` = 0.04 D (the source's 0.02 display threshold for naming a cast, at a mid gray) or when the survivals do not act (`crossover_strength`). The table acts only under color model **Hybrid**. |
| UI | **Crossover** slider (0–100 %) under Estimate, grayed out until the frame has a table and the model is Hybrid. One hint line names the crossover in D and the shadow and highlight hues (a port of `castName`). A closed **Compare** fold holds three `SegmentedChoice` controls. All four controls are on the `fade` roll card, with three REGISTRY cycle actions (no default key), guide bullets and all four catalogs. |
| A/B | `fade_ab_neutrals` (`negpy`/`nc`), `fade_ab_model` (`negpy`/`nc`/`hybrid`), `fade_ab_tone` (`off`/`nc`). The port is in `negpy/features/process/nc_rescue.py`. |

**My reading of the brief, to confirm:** model `negpy` is the Survival sliders alone and ignores the table, even at Crossover > 0; `hybrid` applies the table at the Crossover strength. So the slider is live only under Hybrid. Flip `crossover_strength` in `fade.py` if NegPy should include the slider.

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
| NegPy | negpy | 35.4 | 9.1 | 13.2 | 34.6 |
| Hybrid | negpy | 5.5 | 1.2 | **5.4** | **8.5** |
| NC model | negpy | **5.4** | **1.1** | 10.0 | 23.8 |
| NegPy + NC tone | negpy | 32.1 | 9.7 | 21.3 | 30.7 |
| Hybrid + NC tone | negpy | 3.3 | 1.4 | 19.0 | 26.3 |
| NC model + NC tone | negpy | 4.6 | 1.4 | 24.8 | 36.6 |
| NegPy | nc | 40.4 | 10.3 | 15.5 | 37.7 |
| Hybrid | nc | 8.4 | 2.1 | 5.8 | 11.7 |
| NegPy + NC tone | nc | 38.2 | 10.3 | 22.5 | 35.6 |
| Hybrid + NC tone | nc | 7.4 | 2.0 | 19.6 | 26.8 |

**Linear fade (survivals exact):**

| variant | neutrals | ramp C\* max | ramp C\* mean | patch ΔE mean | patch ΔE max |
|---|---|---|---|---|---|
| NegPy / Hybrid | negpy | 0.6 | 0.2 | 0.3 | 0.6 |
| NC model | negpy | 3.4 | 1.5 | 20.8 | 29.8 |
| NegPy / Hybrid + NC tone | negpy | 0.7 | 0.2 | 16.5 | 20.9 |
| NC model + NC tone | negpy | 3.6 | 1.1 | 24.2 | 38.1 |
| NegPy / Hybrid | nc | 1.1 | 0.3 | 0.6 | 1.1 |

What the numbers say:

- **Neutralizing the grays**: the NC color table and Hybrid tie on this film (ramp max 5.4 vs 5.5).
- **Keeping colors right**: Hybrid wins clearly (patch ΔE 5.4 vs 10.0). NC works on the display positive with the survivals off, so it fixes the grays' hue but not the lost dye density. Saturated colors stay off, and on the linear fade NC is far worse (20.8 vs 0.3).
- **NC tone** moves every variant away from the unfaded film (ΔE +8 to +16). It is a look, levels and S-curve, not a restoration, and it does not undo a crossover by itself (NegPy + NC tone keeps C\* 32).
- **NC neutrals** give slightly worse survivals here than NegPy's neutral axis (Hybrid ramp max 8.4 vs 5.5). Their band refs pull the shadow ref from bands 1–2, where this film's crossover is largest.
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
- **Cast Removal.** It meters the grid after the table, so it fits what the table leaves. Under Hybrid at full strength the neutral axis is near neutral and Cast Removal has little to do. A slide starts with Cast Removal at 0.
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

For each input it writes `<name>.crossover-0.png`, `-0.5.png` and `-1.png` (Hybrid at those strengths) and `<name>.compare.png`. The contact sheet has tone Off and NC across, and NegPy, Hybrid, NC model and NegPy with NC neutrals down. The script prints both Estimates, with the table's zero, its crossover and the band table. Options: `--long-edge 2000`, `--method midtone|spread`, `--gpu` (the NC rows still render on the CPU).
