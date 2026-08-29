# Fade restoration profile gallery

Community-contributed dye-fade parameters for NegPy's **Fade Restoration**
control (Process panel, E-6 only).

Every `.toml` here is bundled with the app and copied into a user's
`<Documents>/NegPy/fade/` folder on first run, so they show up in the sidebar
dropdown out of the box.

A profile is `delta`: the six side-absorption ratios between layers, in
`(gr, br, rg, bg, rb, gb)` order — a property of the dye set, not of any one
faded frame. The two surviving-dye ratios that *do* vary per frame (relative
green/blue survival against red) are not profile data; they live as sliders
in the sidebar, or from the Estimate action, next to Strength. `Generic E6`
ships with `delta` all zero, so the control is visibly present and inert
until real numbers exist for a stock.

```toml
process = "Transparency"   # or "Color Negative", once a negative dye set exists
```

Sourcing real `delta` values means dye spectral-density curves per stock
family. Measured profiles (fit against real faded and unfaded scans of the
same stock) are more useful here than spec-sheet estimates, since the
numbers are the entire correction.
