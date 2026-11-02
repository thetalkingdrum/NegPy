// Runs NegativeConverter's expiredRescue.js on the port-check inputs and prints the results as JSON.
//   node run_nc.mjs <path to expiredRescue.js> <inputs.json>
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';

const [modulePath, inputPath] = process.argv.slice(2);
const nc = await import(pathToFileURL(modulePath).href);
const inputs = JSON.parse(readFileSync(inputPath, 'utf8'));
const out = {};
for (const [name, item] of Object.entries(inputs.cases)) {
  const { width, height } = item;
  // A small Buffer sits at an offset inside Node's shared pool, so slice its own bytes out.
  const bytes = Buffer.from(item.data, 'base64');
  const rgb = new Uint16Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.length));
  const data = new Uint16Array(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    data[i * 4] = rgb[i * 3]; data[i * 4 + 1] = rgb[i * 3 + 1]; data[i * 4 + 2] = rgb[i * 3 + 2]; data[i * 4 + 3] = 65535;
  }
  const analysis = nc.analyzeExpiredFilm({ width, height, data });
  const defaults = nc.defaultExpiredRescueParams(analysis);
  const variants = {
    colour: { expiredLevels: 0, expiredNeutralize: 100, expiredCrossover: 100, expiredBrightness: 0, expiredContrast: 0 },
    tone: { ...defaults, expiredNeutralize: 0, expiredCrossover: 0 },
    both: { ...defaults },
  };
  const result = { analysis, defaults, variants: {} };
  for (const [vname, params] of Object.entries(variants)) {
    const stages = nc.buildExpiredRescueStages({ ...params, expiredEnabled: true, expiredAnalysis: analysis });
    const probes = [];
    for (const [y, x] of inputs.probes) {
      const i = (y * width + x) * 3;
      const px = new Float32Array([rgb[i] * 255 / 65535, rgb[i + 1] * 255 / 65535, rgb[i + 2] * 255 / 65535]);
      nc.applyExpiredTone(stages, px);
      probes.push(Array.from(px));
    }
    result.variants[vname] = {
      offsets: stages.offsets ? Array.from(stages.offsets) : null,
      tone: Array.from(stages.tone),
      probes,
    };
  }
  out[name] = result;
}
process.stdout.write(JSON.stringify(out));
