// Runs NegativeConverter's uneven-fog stage (expiredRescueOpenCv.js with opencv-js, then expiredRescue.js)
// on the fog port-check inputs and prints the results as JSON.
//   node run_nc_fog.mjs <negative2positive/src> <node_modules with @techstark/opencv-js> <inputs.json>
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

const [srcDir, modulesDir, inputPath] = process.argv.slice(2);
const require = createRequire(join(modulesDir, 'noop.js'));
let cv = require('@techstark/opencv-js');
if (cv && typeof cv.then === 'function') cv = await cv;
if (cv && !cv.Mat && cv.default) cv = cv.default;
globalThis.cv = cv;
const nc = await import(pathToFileURL(join(srcDir, 'pipeline', 'expiredRescue.js')).href);
const ncv = await import(pathToFileURL(join(srcDir, 'app', 'expiredRescueOpenCv.js')).href);
const inputs = JSON.parse(readFileSync(inputPath, 'utf8'));
const out = {};
for (const [name, item] of Object.entries(inputs.cases)) {
  const { width, height } = item;
  const bytes = Buffer.from(item.data, 'base64');
  const rgb = new Uint16Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.length));
  const data = new Uint16Array(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    data[i * 4] = rgb[i * 3]; data[i * 4 + 1] = rgb[i * 3 + 1]; data[i * 4 + 2] = rgb[i * 3 + 2]; data[i * 4 + 3] = 65535;
  }
  const image = { width, height, data };
  const maps = ncv.measureExpiredSpatialMaps(image, {});
  const spatial = nc.fitExpiredSpatial(maps);
  const settings = { ...nc.EXPIRED_RESCUE_DEFAULTS, expiredEnabled: true, expiredUnevenFog: 100, expiredLocalContrast: 0, expiredAnalysis: { spatial } };
  const stage = nc.buildExpiredSpatialStage(settings);
  const analysis = nc.analyzeExpiredFilm(image, { spatial: stage });
  const defaults = analysis ? nc.defaultExpiredRescueParams(analysis) : null;
  const result = { maps: { low: maps.low.map((g) => Array.from(g)), gridWidth: maps.gridWidth, gridHeight: maps.gridHeight, fraction: maps.fraction }, spatial, analysis, variants: {} };
  for (const vname of inputs.variants) {
    const stages = vname === 'fog_both' && analysis
      ? nc.buildExpiredRescueStages({ ...defaults, expiredEnabled: true, expiredAnalysis: analysis })
      : null;
    const probes = [];
    for (const [y, x] of inputs.probes) {
      const i = (y * width + x) * 3;
      const px = new Float32Array([rgb[i] * 255 / 65535, rgb[i + 1] * 255 / 65535, rgb[i + 2] * 255 / 65535]);
      if (stage) nc.applyExpiredSpatial(stage, (x + 0.5) / width, (y + 0.5) / height, px);
      if (stages) nc.applyExpiredTone(stages, px);
      probes.push(Array.from(px));
    }
    result.variants[vname] = { probes };
  }
  out[name] = result;
}
process.stdout.write(JSON.stringify(out));
