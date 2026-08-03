/**
 * Tests for the PaulStretch engine.
 *
 * The `Swift parity` block is the important one: the fingerprints are RMS
 * values of 32 equal slices of the Swift reference render (swift-paul-stretch)
 * for a fixed source, seed and parameters. Measured null depth against the full
 * Swift output is -130 dB (correlation 1.000000), i.e. the two implementations
 * agree to float32 rounding. Any real algorithmic drift moves these RMS values
 * well outside the tolerance; harmless rounding never will.
 *
 * `wrapInput` is not reachable from Swift's public API, so it cannot be
 * fingerprinted; it is covered behaviourally instead.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { stretch, DEFAULT_SEED } from '../src/stretch.js';
import { createBuffer, fromChannels } from '../src/buffer.js';

const SR = 44100;

/**
 * The three-tone source both parity suites render. Left is the sum, right is
 * the same at 0.8 — enough stereo difference that a channel mix-up shows up.
 */
function makeSource(seconds, sampleRate = SR) {
  const n = Math.round(seconds * sampleRate);
  const l = new Float32Array(n);
  const r = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const t = i / sampleRate;
    const v = 0.5 * Math.sin(2 * Math.PI * 220 * t)
            + 0.3 * Math.sin(2 * Math.PI * 437 * t)
            + 0.2 * Math.sin(2 * Math.PI * 1103 * t);
    l[i] = v;
    r[i] = v * 0.8;
  }
  return fromChannels(l, r, sampleRate);
}

/** RMS of 32 equal slices of a channel, the shape the fingerprints are in. */
function sliceRms(ch, slices = 32) {
  const size = Math.ceil(ch.length / slices);
  const out = [];
  for (let s = 0; s < slices; s++) {
    const slice = ch.subarray(s * size, Math.min(ch.length, (s + 1) * size));
    let acc = 0;
    for (let i = 0; i < slice.length; i++) acc += slice[i] * slice[i];
    out.push(Math.sqrt(acc / slice.length));
  }
  return out;
}

/** RMS over the fractional span [from, to) of a channel. */
function rmsSpan(ch, from, to) {
  const a = Math.floor(ch.length * from);
  const b = Math.floor(ch.length * to);
  let acc = 0;
  for (let i = a; i < b; i++) acc += ch[i] * ch[i];
  return Math.sqrt(acc / (b - a));
}

/** Number of samples that differ between two buffers, across both channels. */
function differingSamples(a, b) {
  let differing = 0;
  for (const ch of ['l', 'r']) {
    const x = a[ch];
    const y = b[ch];
    if (x.length !== y.length) {
      differing += Math.abs(x.length - y.length);
      continue;
    }
    for (let i = 0; i < x.length; i++) if (x[i] !== y[i]) differing++;
  }
  return differing;
}

// ── Swift parity ────────────────────────────

/**
 * RMS of 32 equal slices of the Swift reference render.
 *
 * `pitch` was RE-CAPTURED after pitch shifting moved from FFT-bin remapping to
 * resampling the input (Catmull-Rom + a decimation pre-filter, stretch ratio
 * rescaled so the output length is unchanged). The old fingerprint described an
 * algorithm neither side runs any more. `pitchDown` covers the resampler's
 * other direction — interpolation rather than decimation, so it exercises the
 * branch that skips the pre-filter.
 *
 * Note that `drift` is a weak discriminator on this particular source: three
 * steady sine tones are spectrally stationary, so moving the analysis read head
 * barely changes the magnitudes it sees. It still pins the read-position
 * arithmetic — the divergence from `full` shows up in the last quarter, where a
 * drifted read head runs off the end of the input at different windows.
 */
const SWIFT = {
  full: [0.241204378, 0.336966745, 0.323215669, 0.262875124, 0.387490231, 0.299155869, 0.402720786, 0.265028547, 0.296225584, 0.345915491, 0.298012395, 0.239222073, 0.203054672, 0.316449671, 0.302950430, 0.270245087, 0.287904073, 0.252276596, 0.200117776, 0.269020032, 0.205544377, 0.227606229, 0.298868020, 0.203596649, 0.334314572, 0.277291647, 0.302229865, 0.185383417, 0.237660128, 0.233788681, 0.151549462, 0.044584452],
  half: [0.159339998, 0.306398167, 0.268658611, 0.265026434, 0.245967760, 0.285245683, 0.299699124, 0.258881504, 0.256540615, 0.293423530, 0.282191809, 0.267337095, 0.221936945, 0.256698046, 0.227563273, 0.219562416, 0.246281333, 0.252066028, 0.353061290, 0.229326136, 0.307392397, 0.332142041, 0.234391846, 0.248134539, 0.229423560, 0.284330026, 0.301875734, 0.215605259, 0.291071069, 0.265852521, 0.142692216, 0.030091002],
  pitch: [0.161818806, 0.229722420, 0.236140321, 0.231964914, 0.186862748, 0.265816809, 0.303135995, 0.202435798, 0.208687239, 0.230094525, 0.272242329, 0.297689520, 0.232757564, 0.202563220, 0.231418445, 0.197278963, 0.246076376, 0.249967863, 0.295677826, 0.267285906, 0.245883848, 0.331210674, 0.202475848, 0.179524900, 0.268026369, 0.277344090, 0.300448283, 0.187771055, 0.143296979, 0.079675937, 0.049436756, 0.019944822],
  pitchDown: [0.118338660, 0.268829689, 0.220455956, 0.256281434, 0.265496289, 0.272843030, 0.178904226, 0.392150305, 0.260766287, 0.261847950, 0.317387933, 0.281380654, 0.147502231, 0.279547459, 0.237795209, 0.214003861, 0.266251342, 0.277322426, 0.199699423, 0.216676186, 0.205159889, 0.207869699, 0.253324576, 0.311307643, 0.255810754, 0.308416021, 0.194572924, 0.222139816, 0.333494803, 0.251599290, 0.189921494, 0.135563745],
  drift: [0.241204102, 0.336966232, 0.323215260, 0.262874760, 0.387489637, 0.299155554, 0.402720220, 0.265028145, 0.296225128, 0.345915000, 0.298012060, 0.239221652, 0.203054385, 0.316449219, 0.302949956, 0.270244748, 0.287903726, 0.252276293, 0.200117444, 0.269019644, 0.205544082, 0.227605904, 0.298867658, 0.203630999, 0.320066586, 0.244872372, 0.265403662, 0.202631943, 0.232228167, 0.233635294, 0.152463431, 0.057070135],
  continuity: [0.160273915, 0.258436407, 0.263824622, 0.177847869, 0.251487552, 0.185176842, 0.363880481, 0.235828534, 0.243761013, 0.215241508, 0.170117777, 0.227479579, 0.167608238, 0.210276529, 0.167177634, 0.232837128, 0.166121020, 0.155756485, 0.151110302, 0.219920963, 0.187678588, 0.173687975, 0.165659477, 0.129666281, 0.251916351, 0.240050034, 0.225649759, 0.154228458, 0.177875136, 0.195611903, 0.125393847, 0.035917165],
};

/** Per-case overrides layered onto the fixed `{ windowSec: 0.25 }` render. */
const CASES = [
  ['full',       { phaseRandomness: 1,   pitchSemitones: 0 }],
  ['half',       { phaseRandomness: 0.5, pitchSemitones: 0 }],
  ['pitch',      { phaseRandomness: 1,   pitchSemitones: 7 }],
  ['pitchDown',  { phaseRandomness: 1,   pitchSemitones: -7 }],
  ['drift',      { phaseRandomness: 1,   pitchSemitones: 0, spectralDrift: 0.5 }],
  ['continuity', { phaseRandomness: 1,   pitchSemitones: 0, phaseContinuity: 0.3 }],
];

const TOL = 1e-5;

describe('stretch — Swift parity', () => {
  const source = makeSource(2);

  for (const [name, opts] of CASES) {
    it(`${name} matches the Swift reference render`, async () => {
      const out = await stretch(source, 4, { windowSec: 0.25, ...opts });
      const got = sliceRms(out.l);
      const want = SWIFT[name];
      let worst = 0;
      for (let s = 0; s < 32; s++) worst = Math.max(worst, Math.abs(got[s] - want[s]));
      assert.ok(worst < TOL, `worst slice RMS delta ${worst.toExponential(2)}, tol ${TOL}`);
    });
  }

  it('leaves the source untouched', async () => {
    const src = makeSource(0.2);
    const before = Float32Array.from(src.l);
    await stretch(src, 4, { windowSec: 0.05 });
    assert.deepEqual(src.l, before);
  });
});

// ── wrapInput ───────────────────────────────

/**
 * `wrapInput` cannot be fingerprinted against Swift — `StretchKernel.wrapInput`
 * exists there but is not surfaced by `PaulStretcher.stretch`, so there is no
 * public entry point that turns it on and no reference render to pin. It is
 * tested by what it is for instead.
 *
 * What it is for: the analysis read head advances through the input as the
 * render proceeds, and the LAST windows start close enough to the end that most
 * of the window falls past it. Zero-padding those samples makes the output
 * decay to silence over the final `windowSize / inputLen` fraction — negligible
 * for a long source, ruinous for a short one, and exactly the material a
 * seamless-loop crossfade would splice onto a full-energy head.
 *
 * The source is deliberately short relative to the window (1 s source, 0.25 s
 * window, ratio 30) so the decay covers roughly the last third.
 *
 * Thresholds were set from measurement, not guessed:
 *   unwrapped end/mid = 0.1233 (deciles ... 0.2274 0.1400 0.0289)
 *   wrapped   end/mid = 1.0119 (deciles ... 0.2479 0.2433 0.2372)
 * so 0.35 and 0.75..1.35 both sit ~3× clear of what was observed.
 */
describe('stretch — wrapInput', () => {
  const RATIO = 30;
  const opts = { windowSec: 0.25 };
  /** Ceiling on the unwrapped render's end/mid RMS ratio (measured 0.1233). */
  const MAX_UNWRAPPED_TAIL = 0.35;
  /** Band the wrapped render's end/mid RMS ratio must stay inside (measured 1.0119). */
  const MIN_WRAPPED_TAIL = 0.75;
  const MAX_WRAPPED_TAIL = 1.35;

  const src = makeSource(1);

  it('unwrapped render decays over its final 10 %', async () => {
    const out = await stretch(src, RATIO, { ...opts, wrapInput: false });
    const ratio = rmsSpan(out.l, 0.90, 1.00) / rmsSpan(out.l, 0.45, 0.55);
    assert.ok(ratio < MAX_UNWRAPPED_TAIL,
      `end/mid RMS ${ratio.toFixed(4)} should be < ${MAX_UNWRAPPED_TAIL}`);
  });

  it('wrapped render sustains to its final sample', async () => {
    const out = await stretch(src, RATIO, { ...opts, wrapInput: true });
    const ratio = rmsSpan(out.l, 0.90, 1.00) / rmsSpan(out.l, 0.45, 0.55);
    assert.ok(ratio > MIN_WRAPPED_TAIL && ratio < MAX_WRAPPED_TAIL,
      `end/mid RMS ${ratio.toFixed(4)} should be in ${MIN_WRAPPED_TAIL}..${MAX_WRAPPED_TAIL}`);
  });

  /**
   * Pins the DEFAULT path rather than comparing wrapped to unwrapped on a long
   * source: the two are never bit-identical, because the read head runs off the
   * end of ANY source in its final windows. What IS exactly true is that
   * `wrapInput: false` renders the same samples as omitting the option — the
   * property that matters, since it means existing renders are unchanged.
   */
  it('wrapInput: false is bit-identical to omitting the option', async () => {
    const unwrapped = await stretch(src, RATIO, { ...opts, wrapInput: false });
    const omitted = await stretch(src, RATIO, opts);
    assert.equal(differingSamples(unwrapped, omitted), 0);
  });

  it('wraps from a negative read position without reading out of bounds', async () => {
    // Drift can pull the read head before frame 0 early in the render; the
    // floor-mod has to bring it back inside the source rather than produce a
    // negative index (undefined → NaN through the whole window).
    const out = await stretch(makeSource(0.2), 8, {
      windowSec: 0.05,
      spectralDrift: 1,
      wrapInput: true,
    });
    assert.ok(out.l.every(Number.isFinite), 'no NaN or Infinity in the output');
  });
});

// ── Passthrough ─────────────────────────────

describe('stretch — passthrough', () => {
  it('returns the input unchanged at ratio 1', async () => {
    const src = makeSource(0.05);
    const out = await stretch(src, 1);
    assert.deepEqual(out.l, src.l);
    assert.deepEqual(out.r, src.r);
    assert.equal(out.sampleRate, src.sampleRate);
  });

  it('passes through at the 1.001 boundary but stretches just above it', async () => {
    const src = makeSource(0.05);
    const at = await stretch(src, 1.001);
    assert.equal(at.l.length, src.l.length);
    const above = await stretch(src, 1.002, { windowSec: 0.01 });
    assert.notEqual(above.l.length, src.l.length);
  });

  it('returns a copy, not the source arrays', async () => {
    const src = makeSource(0.05);
    const out = await stretch(src, 1);
    assert.notEqual(out.l, src.l);
    assert.notEqual(out.r, src.r);
    out.l[0] = 123;
    assert.notEqual(src.l[0], 123);
  });

  it('splits a shared mono array into two independent channels', async () => {
    const mono = new Float32Array([0.1, -0.2, 0.3, -0.4]);
    const out = await stretch(fromChannels(mono, undefined, SR), 1);
    assert.notEqual(out.l, out.r);
    assert.deepEqual(out.l, out.r);
  });
});

// ── Determinism ─────────────────────────────

describe('stretch — determinism', () => {
  const src = makeSource(0.25);
  const opts = { windowSec: 0.05, spectralDrift: 0.4, phaseContinuity: 0.3 };

  it('the same seed renders bit-identical output', async () => {
    const a = await stretch(src, 6, { ...opts, seed: 0xabcdef0123456789n });
    const b = await stretch(src, 6, { ...opts, seed: 0xabcdef0123456789n });
    assert.equal(differingSamples(a, b), 0);
  });

  it('omitting the seed is the same as passing DEFAULT_SEED', async () => {
    const a = await stretch(src, 6, opts);
    const b = await stretch(src, 6, { ...opts, seed: DEFAULT_SEED });
    assert.equal(differingSamples(a, b), 0);
  });

  it('a different seed renders different output of the same length', async () => {
    const a = await stretch(src, 6, { ...opts, seed: DEFAULT_SEED });
    const b = await stretch(src, 6, { ...opts, seed: DEFAULT_SEED + 1n });
    assert.equal(a.l.length, b.l.length);
    assert.ok(differingSamples(a, b) > a.l.length, 'most samples differ');
  });

  it('DEFAULT_SEED matches PaulStretcher.defaultSeed', () => {
    assert.equal(DEFAULT_SEED, 0x2545f4914f6cdd1dn);
  });
});

// ── Parameters ──────────────────────────────

describe('stretch — parameters', () => {
  const src = makeSource(0.25);

  it('output length is ratio × input, and pitch shifting does not change it', async () => {
    const plain = await stretch(src, 5, { windowSec: 0.05 });
    assert.equal(plain.l.length, src.l.length * 5);
    // The stretch ratio is rescaled by the pitch factor so the resampled
    // source still lands on the same output length (± the resampler's own
    // rounding, which is a handful of frames on a 55 k buffer).
    const up = await stretch(src, 5, { windowSec: 0.05, pitchSemitones: 7 });
    const down = await stretch(src, 5, { windowSec: 0.05, pitchSemitones: -7 });
    assert.ok(Math.abs(up.l.length - plain.l.length) < 32, `up ${up.l.length} vs ${plain.l.length}`);
    assert.ok(Math.abs(down.l.length - plain.l.length) < 32, `down ${down.l.length} vs ${plain.l.length}`);
  });

  it('output is at least one window long even for a very short source', async () => {
    const tiny = fromChannels(new Float32Array([0.5, -0.5, 0.25, -0.25]), undefined, SR);
    const out = await stretch(tiny, 4, { windowSec: 0.01 });
    // windowSec 0.01 at 44.1 kHz → trunc(441) → nextPow2 → 512.
    assert.equal(out.l.length, 512);
  });

  it('preserves the sample rate', async () => {
    const out = await stretch(makeSource(0.1, 8000), 4, { windowSec: 0.05 });
    assert.equal(out.sampleRate, 8000);
  });

  it('peak-normalises to 0.92', async () => {
    const out = await stretch(src, 4, { windowSec: 0.05 });
    let peak = 0;
    for (let i = 0; i < out.l.length; i++) {
      peak = Math.max(peak, Math.abs(out.l[i]), Math.abs(out.r[i]));
    }
    assert.ok(Math.abs(peak - 0.92) < 1e-6, `peak ${peak}`);
  });

  it('phaseRandomness 0 leaves the source phase alone, so no seed matters', async () => {
    const opts = { windowSec: 0.05, phaseRandomness: 0 };
    const a = await stretch(src, 4, { ...opts, seed: 1n });
    const b = await stretch(src, 4, { ...opts, seed: 999n });
    assert.equal(differingSamples(a, b), 0);
  });

  it('clamps out-of-range 0..1 parameters instead of misbehaving', async () => {
    const opts = { windowSec: 0.05 };
    const clamped = await stretch(src, 4, { ...opts, phaseRandomness: 5, spectralDrift: -1 });
    const explicit = await stretch(src, 4, { ...opts, phaseRandomness: 1, spectralDrift: 0 });
    assert.equal(differingSamples(clamped, explicit), 0);
  });

  it('onsetSensitivity changes the render, and 0 is the classic path', async () => {
    const opts = { windowSec: 0.05 };
    const off = await stretch(src, 4, opts);
    const explicitOff = await stretch(src, 4, { ...opts, onsetSensitivity: 0 });
    assert.equal(differingSamples(off, explicitOff), 0);
    const on = await stretch(src, 4, { ...opts, onsetSensitivity: 1 });
    assert.ok(differingSamples(off, on) > 0, 'onset protection alters the output');
  });

  it('phaseContinuity is inert below full phaseRandomness', async () => {
    // Partial randomisation is already anchored to the source phase, so the
    // continuity draw is discarded — but it must still happen, keeping the two
    // streams in lockstep and the output identical either way.
    const opts = { windowSec: 0.05, phaseRandomness: 0.5 };
    const a = await stretch(src, 4, opts);
    const b = await stretch(src, 4, { ...opts, phaseContinuity: 0.9 });
    assert.equal(differingSamples(a, b), 0);
  });

  it('widens a mono source into a decorrelated stereo wash', async () => {
    // L and R draw from the SAME per-window generator, L first, so a mono
    // source comes out with independent phases in each channel: the wash is
    // stereo even though the input was not. This is the Swift kernel's
    // consumption order, not an accident of the port.
    const mono = fromChannels(makeSource(0.25).l, undefined, SR);
    const out = await stretch(mono, 4, { windowSec: 0.05 });
    assert.ok(out.l.some((v) => v !== 0));
    let identical = 0;
    for (let i = 0; i < out.l.length; i++) if (out.l[i] === out.r[i]) identical++;
    assert.ok(identical < out.l.length / 100, `${identical} of ${out.l.length} samples identical`);
  });

  it('resamples a shared mono array once when pitch shifting', async () => {
    const mono = fromChannels(makeSource(0.25).l, undefined, SR);
    const out = await stretch(mono, 4, { windowSec: 0.05, pitchSemitones: 5 });
    assert.ok(out.l.every(Number.isFinite));
    assert.ok(out.l.some((v) => v !== 0));
  });
});

// ── Edge cases ──────────────────────────────

describe('stretch — edge cases', () => {
  it('renders a silent window from an empty input rather than throwing', async () => {
    const empty = createBuffer(0, SR);
    const out = await stretch(empty, 4, { windowSec: 0.01 });
    assert.equal(out.l.length, 512);
    assert.ok(out.l.every((v) => v === 0));
  });

  it('passes an empty input straight through at ratio 1', async () => {
    const out = await stretch(createBuffer(0, SR), 1);
    assert.equal(out.l.length, 0);
  });

  it('leaves silence silent (no divide-by-zero in the normalise)', async () => {
    const out = await stretch(createBuffer(4096, SR), 4, { windowSec: 0.05 });
    assert.ok(out.l.every((v) => v === 0));
    assert.ok(out.r.every((v) => v === 0));
  });

  it('skips the resampler for a source too short to interpolate', async () => {
    // Fewer than four frames: Swift's `resampled(_:by:)` bails, so the pitch
    // shift silently becomes a no-op on the source (the ratio is still
    // rescaled, which is what makes the two lengths differ).
    const three = fromChannels(new Float32Array([0.5, -0.5, 0.5]), undefined, SR);
    const out = await stretch(three, 4, { windowSec: 0.01, pitchSemitones: 12 });
    assert.equal(out.l.length, 512);
    assert.ok(out.l.every(Number.isFinite));
  });

  it('produces finite output at extreme stretch ratios', async () => {
    const out = await stretch(makeSource(0.05), 200, { windowSec: 0.02 });
    assert.ok(out.l.every(Number.isFinite));
    assert.ok(out.l.some((v) => v !== 0));
  });
});

// ── Progress ────────────────────────────────

describe('stretch — progress', () => {
  const src = makeSource(0.5);

  it('reports a non-decreasing fraction ending at 1', async () => {
    const seen = [];
    await stretch(src, 8, { windowSec: 0.02, onProgress: (f) => { seen.push(f); } });
    assert.ok(seen.length > 1, `expected several progress calls, got ${seen.length}`);
    assert.equal(seen.at(-1), 1);
    for (let i = 1; i < seen.length; i++) assert.ok(seen[i] >= seen[i - 1]);
    assert.ok(seen[0] >= 0 && seen[0] <= 1);
  });

  it('awaits whatever onProgress returns', async () => {
    let resolved = 0;
    await stretch(src, 8, {
      windowSec: 0.02,
      onProgress: () => new Promise((r) => { resolved++; r(); }),
    });
    assert.ok(resolved > 1);
  });

  it('reports progress on the passthrough path too', async () => {
    const seen = [];
    await stretch(src, 1, { onProgress: (f) => { seen.push(f); } });
    assert.deepEqual(seen, [1]);
  });

  it('renders identically with and without a progress hook', async () => {
    const opts = { windowSec: 0.02 };
    const quiet = await stretch(src, 8, opts);
    const noisy = await stretch(src, 8, { ...opts, onProgress: () => {} });
    assert.equal(differingSamples(quiet, noisy), 0);
  });
});
