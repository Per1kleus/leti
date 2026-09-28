// Measures the curve gui/hud.html actually draws, in every state.
//
// The functions are lifted out of the HUD by brace matching rather than copied,
// so this cannot drift away from what a browser runs: rename or remove one and
// this fails instead of quietly measuring a stale duplicate. Run by
// tests/test_hud_core_geometry.py, which skips it when node is not installed.
//
// The number it prints, WORST, is the sharpest turn anywhere on the densely
// sampled outline, in degrees. A perfect circle measures 0.32 at this sampling; a
// seam between the last point and the first shows up as a large one.
'use strict';

const fs = require('fs');
const html = fs.readFileSync(process.argv[2] || 'gui/hud.html', 'utf8');

function grab(name){
  const start = html.indexOf('function ' + name + '(');
  if(start < 0) throw new Error('gui/hud.html no longer has a function called ' + name);
  let depth = 0;
  for(let j = html.indexOf('{', start); j < html.length; j++){
    if(html[j] === '{') depth++;
    else if(html[j] === '}'){ depth--; if(depth === 0) return html.slice(start, j + 1); }
  }
  throw new Error('unbalanced braces in ' + name);
}
function num(name){
  const m = html.match(new RegExp('\\b' + name + '\\s*=\\s*(\\d+)'));
  if(!m) throw new Error('gui/hud.html no longer defines ' + name);
  return Number(m[1]);
}
function expr(name){
  const m = html.match(new RegExp('^\\s*const ' + name + '\\s*=\\s*([^;]+);', 'm'));
  if(!m) throw new Error('gui/hud.html no longer defines ' + name);
  return m[1].trim();
}

const source = [
  'const BINS = ' + num('BINS') + ';',
  'const BASE_R = ' + num('BASE_R') + ';',
  'const AMP_R = ' + num('AMP_R') + ';',
  'const CENTER = ' + num('CENTER') + ';',
  'const HALF_BINS = BINS / 2;',
  'const NOISE_ANCHORS = ' + num('NOISE_ANCHORS') + ';',
  'const RING_SMOOTHING_PASSES = ' + num('RING_SMOOTHING_PASSES') + ';',
  'const IDLE_WAVES = ' + num('IDLE_WAVES') + ';',
  'const ALERT_WAVES = ' + num('ALERT_WAVES') + ';',
  'const SPEAK_WAVES = ' + num('SPEAK_WAVES') + ';',
  'const SPEAK_RIPPLE = ' + num('SPEAK_RIPPLE') + ';',
  'const ringHarmonic = ' + expr('ringHarmonic') + ';',
  'const ringScratch = new Float32Array(BINS);',
  'const noiseFrom = new Float32Array(BINS);',
  'const noiseTo = new Float32Array(BINS);',
  'const noiseAnchors = new Float32Array(NOISE_ANCHORS);',
  'let noisePhase = 0;',
  'const target = new Float32Array(BINS);',
  "let clock = 0, mode = 'idle', analyser = null, dataArray = null;",
  grab('smoothRing'), grab('fillRingNoise'), grab('advanceNoise'),
  grab('noiseAt'), grab('updateTargets'), grab('smoothClosedPath'),
  'module.exports = {',
  '  BINS, BASE_R, AMP_R, CENTER, target, smoothClosedPath, updateTargets,',
  '  fillRingNoise, advanceNoise, noiseFrom, noiseTo,',
  '  set: (m, c, a, d) => { mode = m; clock = c; analyser = a; dataArray = d; },',
  '};',
].join('\n');

const hud = new Function('module', 'exports', source);
const mod = { exports: {} };
hud(mod, mod.exports);
const H = mod.exports;
H.fillRingNoise(H.noiseFrom);
H.fillRingNoise(H.noiseTo);

function pointsOf(amps){
  const pts = [];
  for(let i = 0; i < H.BINS; i++){
    const a = (i / H.BINS) * Math.PI * 2 - Math.PI / 2;
    const r = H.BASE_R + amps[i] * H.AMP_R;
    pts.push([H.CENTER + r * Math.cos(a), H.CENTER + r * Math.sin(a)]);
  }
  return pts;
}

function samplePath(d, per){
  per = per || 24;
  // Parsed back out of the very string the HUD hands the browser, so a malformed
  // path fails here rather than being drawn as nothing.
  const nums = d.replace(/[MCZ,]/g, ' ').trim().split(/\s+/).map(Number);
  if(nums.some(Number.isNaN)) throw new Error('the path string contains a non-number');
  const start = [nums[0], nums[1]];
  const out = [];
  let cur = start;
  for(let k = 2; k + 5 < nums.length; k += 6){
    const s = [cur, [nums[k], nums[k+1]], [nums[k+2], nums[k+3]], [nums[k+4], nums[k+5]]];
    for(let j = 0; j < per; j++){
      const t = j / per, u = 1 - t;
      const b = [u*u*u, 3*u*u*t, 3*u*t*t, t*t*t];
      out.push([b[0]*s[0][0] + b[1]*s[1][0] + b[2]*s[2][0] + b[3]*s[3][0],
                b[0]*s[0][1] + b[1]*s[1][1] + b[2]*s[2][1] + b[3]*s[3][1]]);
    }
    cur = s[3];
  }
  if(Math.hypot(cur[0] - start[0], cur[1] - start[1]) > 1e-6){
    throw new Error('the path does not come back to where it started');
  }
  return out;
}

function sharpest(poly){
  let worst = 0, where = -1;
  const n = poly.length;
  for(let i = 0; i < n; i++){
    const p = poly[(i - 1 + n) % n], c = poly[i], q = poly[(i + 1) % n];
    const ax = c[0]-p[0], ay = c[1]-p[1], bx = q[0]-c[0], by = q[1]-c[1];
    const na = Math.hypot(ax, ay), nb = Math.hypot(bx, by);
    if(na < 1e-9 || nb < 1e-9) continue;
    const ang = Math.acos(Math.max(-1, Math.min(1, (ax*bx + ay*by) / (na*nb)))) * 180 / Math.PI;
    if(ang > worst){ worst = ang; where = i / n; }
  }
  return [worst, where];
}

let seed = 7;
function rnd(){ seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; }
function speech(n){
  const a = new Uint8Array(n);
  for(let k = 0; k < n; k++){
    const f = k / n;
    a[k] = Math.max(0, Math.min(255, (235 * Math.exp(-f * 9) + 6) * (0.55 + 0.9 * rnd())));
  }
  return a;
}

let worst = 0, worstAt = '';
function probe(label){
  const [w, where] = sharpest(samplePath(H.smoothClosedPath(pointsOf(H.target))));
  console.log(label.padEnd(36) + w.toFixed(2).padStart(6) + ' deg at '
              + (where * 100).toFixed(1).padStart(5) + '% round');
  if(w > worst){ worst = w; worstAt = label; }
}

[0, 0.7, 1.9, 3.0, 4.4].forEach(t => { H.set('idle', t, null, null); H.updateTargets(1/60); probe('idle t=' + t); });
[0, 1.3, 3.0].forEach(t => { H.set('listening', t, null, null); H.updateTargets(1/60); probe('listening, no browser mic t=' + t); });
// Speaking is stepped frame by frame rather than jumped between clock values,
// because the noise field only regenerates once every few frames (advanceNoise
// ticks seven times a second at dt = 1/60) and a probe that never reaches a
// regeneration never measures the field the running HUD actually uses. Found the
// hard way: reverting the anchored field to per-bin randomness left this check
// passing until it stepped through enough frames to refresh it.
for(let f = 0, t = 0; f < 480; f++, t += 1/60){
  H.set('speaking', t, null, null);
  H.updateTargets(1/60);
  if(f % 40 === 0) probe('speaking frame ' + f);
}

const data = new Uint8Array(1024);
const fake = { getByteFrequencyData: a => a.set(speech(a.length)) };
for(let i = 0; i < 8; i++){
  H.set('listening', i * 0.4, fake, data);
  H.updateTargets(1/60);
  probe('listening, real mic ' + (i + 1));
}

// The seam, stated directly: bin 0 and bin BINS-1 are neighbours on the ring.
const seam = Math.abs(H.target[0] - H.target[H.BINS - 1]);
console.log('seam between the last point and the first: '
            + (seam * H.AMP_R).toFixed(3) + ' px of radius');

console.log('WORST ' + worst.toFixed(2) + '  (' + worstAt + ')');
