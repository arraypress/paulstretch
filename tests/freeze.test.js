/**
 * Spectral freeze: cross-implementation parity with `FreezeKernel.swift` /
 * `SpectralFreezer.swift` (swift-paul-stretch), plus behavioural coverage.
 *
 * The fingerprints below are RMS values of 32 equal slices of the SWIFT render
 * for a fixed source, seed and parameters — they are not self-generated, and
 * they are the reason this port can be trusted. Measured null depth against the
 * full Swift output is -131 dB with correlation 1.000000. Do not regenerate
 * them from this implementation; if they fail, this implementation drifted.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { freeze } from '../src/freeze.js';
import { createBuffer, fromChannels, peak } from '../src/buffer.js';
import { DEFAULT_SEED } from '../src/stretch.js';

const SWIFT = {
  plain: [0.113119871, 0.243754853, 0.360196687, 0.241645949, 0.336062860, 0.265262557, 0.199852250, 0.288921432, 0.332777265, 0.204297388, 0.187867744, 0.304517309, 0.240732933, 0.332780888, 0.258730014, 0.201263522, 0.193622024, 0.196910937, 0.243920506, 0.213914934, 0.208873162, 0.235009188, 0.292367507, 0.326108239, 0.319597254, 0.348895307, 0.178990009, 0.240149879, 0.308730242, 0.307608741, 0.381074321, 0.357399186],
  smear: [0.068617823, 0.203405566, 0.212070085, 0.189636261, 0.200960746, 0.212841238, 0.200081444, 0.208348687, 0.193393253, 0.189175984, 0.209904290, 0.217107551, 0.246027850, 0.226545303, 0.217437488, 0.177466921, 0.209215070, 0.206716122, 0.188904951, 0.193488411, 0.209027492, 0.208320634, 0.224438639, 0.208554431, 0.189327161, 0.226295548, 0.212743246, 0.210218039, 0.204491472, 0.195425478, 0.196860894, 0.208276130],
  scan: [0.113120002, 0.243755178, 0.360197119, 0.241646315, 0.336063233, 0.265262946, 0.199852427, 0.288921962, 0.332777804, 0.204297687, 0.187868025, 0.304517646, 0.240733144, 0.332781264, 0.258730384, 0.201263826, 0.193622185, 0.196911222, 0.243920746, 0.213915149, 0.208873565, 0.235009311, 0.292368084, 0.326108662, 0.319597422, 0.348895773, 0.178990028, 0.240150235, 0.308730610, 0.307609137, 0.381074505, 0.357399681],
};

/** `[name, positionNorm, smear, scan]` — the parameter sets the Swift renders used. */
const CASES = [
  ['plain', 0.5, 0, 0],
  ['smear', 0.5, 0.6, 0],
  ['scan', 0.2, 0, 0.8],
];

const TOL = 1e-5;

/** The exact source the Swift fingerprints were captured from. */
function paritySource() {
  const sr = 44100;
  const n = sr * 2;
  const l = new Float32Array(n);
  const r = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const t = i / sr;
    const v = 0.5 * Math.sin(2 * Math.PI * 220 * t)
      + 0.3 * Math.sin(2 * Math.PI * 437 * t)
      + 0.2 * Math.sin(2 * Math.PI * 1103 * t);
    l[i] = v;
    r[i] = v * 0.8;
  }
  return fromChannels(l, r, sr);
}

/** RMS of each of 32 equal slices — the fingerprint the Swift side reports. */
function sliceRms(channel, slices = 32) {
  const size = Math.ceil(channel.length / slices);
  const out = [];
  for (let s = 0; s < slices; s++) {
    const slice = channel.subarray(s * size, Math.min(channel.length, (s + 1) * size));
    let acc = 0;
    for (let i = 0; i < slice.length; i++) acc += slice[i] * slice[i];
    out.push(Math.sqrt(acc / slice.length));
  }
  return out;
}

// ── Small, fast fixtures for the behavioural tests ──────────

const SR = 8000;
const FAST = { windowSec: 0.05 };

/**
 * A source whose spectrum genuinely evolves. A steady tone would give the same
 * magnitudes at every capture point, which would make the scan tests vacuous —
 * as the parity fingerprints above show, `plain` and `scan` are near-identical
 * on a stationary source.
 */
function sweepSource(seconds = 1, sr = SR) {
  const n = Math.trunc(seconds * sr);
  const l = new Float32Array(n);
  const r = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const t = i / sr;
    const f = 120 + 1400 * (i / Math.max(1, n - 1));
    const v = 0.6 * Math.sin(2 * Math.PI * f * t);
    l[i] = v;
    r[i] = v * 0.7;
  }
  return fromChannels(l, r, sr);
}

/** Largest absolute difference between two channels of equal length. */
function maxAbsDiff(a, b) {
  assert.equal(a.length, b.length, 'lengths differ');
  let worst = 0;
  for (let i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs(a[i] - b[i]));
  return worst;
}

function isFiniteChannel(a) {
  for (let i = 0; i < a.length; i++) if (!Number.isFinite(a[i])) return false;
  return true;
}

// ── Parity ──────────────────────────────────────────────────

describe('freeze — Swift parity', () => {
  for (const [name, positionNorm, smear, scan] of CASES) {
    it(`matches the Swift render fingerprint (${name})`, async () => {
      const out = await freeze(paritySource(), 4, { positionNorm, smear, scan, windowSec: 0.25 });
      const got = sliceRms(out.l);
      const want = SWIFT[name];
      let worst = 0;
      for (let s = 0; s < want.length; s++) worst = Math.max(worst, Math.abs(got[s] - want[s]));
      assert.ok(worst < TOL, `freeze/${name} diverges: ${worst.toExponential(2)} > ${TOL}`);
    });
  }
});

// ── Output geometry ─────────────────────────────────────────

describe('freeze — output geometry', () => {
  it('renders trunc(targetSeconds * sampleRate) frames', async () => {
    const out = await freeze(sweepSource(), 1.5, FAST);
    assert.equal(out.l.length, Math.trunc(1.5 * SR));
    assert.equal(out.r.length, out.l.length);
  });

  it('never renders less than one window', async () => {
    // windowSec 0.05 at 8 kHz rounds up to a 512-frame window, which is longer
    // than the 80 frames 0.01 s would give.
    const out = await freeze(sweepSource(), 0.01, FAST);
    assert.equal(out.l.length, 512);
  });

  it('rounds the window up to a power of two', async () => {
    // 0.1 s at 8 kHz is 800 frames, so the window is 1024.
    const out = await freeze(sweepSource(), 0.001, { windowSec: 0.1 });
    assert.equal(out.l.length, 1024);
  });

  it('carries the source sample rate through', async () => {
    const out = await freeze(sweepSource(1, 22050), 0.5, FAST);
    assert.equal(out.sampleRate, 22050);
  });

  it('produces two independent channels', async () => {
    const out = await freeze(sweepSource(), 0.5, FAST);
    assert.notEqual(out.l, out.r);
  });
});

// ── Short and degenerate inputs ─────────────────────────────

describe('freeze — short and degenerate inputs', () => {
  it('renders silence for inputs under 32 frames', async () => {
    const tiny = fromChannels(new Float32Array(31).fill(0.5), undefined, SR);
    const out = await freeze(tiny, 0.5, FAST);
    assert.equal(out.l.length, Math.trunc(0.5 * SR));
    assert.equal(peak(out), 0);
  });

  it('renders silence for an empty input', async () => {
    const out = await freeze(createBuffer(0, SR), 0.5, FAST);
    assert.equal(out.l.length, Math.trunc(0.5 * SR));
    assert.equal(peak(out), 0);
  });

  it('still reports progress on the too-short path', async () => {
    const seen = [];
    const tiny = fromChannels(new Float32Array(10), undefined, SR);
    await freeze(tiny, 0.5, { ...FAST, onProgress: (f) => { seen.push(f); } });
    assert.deepEqual(seen, [1]);
  });

  it('renders audio from exactly 32 frames', async () => {
    // The window is far longer than the source, so the capture is the source
    // followed by zeros — thin, but not silent.
    const l = new Float32Array(32);
    for (let i = 0; i < 32; i++) l[i] = Math.sin(i * 0.7) * 0.5;
    const out = await freeze(fromChannels(l, undefined, SR), 0.25, FAST);
    assert.ok(peak(out) > 0);
    assert.ok(isFiniteChannel(out.l));
  });

  it('leaves a silent source silent instead of dividing by zero', async () => {
    const out = await freeze(createBuffer(SR, SR), 0.5, FAST);
    assert.equal(peak(out), 0);
    assert.ok(isFiniteChannel(out.l));
    assert.ok(isFiniteChannel(out.r));
  });

  it('accepts a mono buffer whose channels share one array', async () => {
    const src = sweepSource();
    const mono = fromChannels(src.l, undefined, SR);
    assert.equal(mono.l, mono.r);
    const out = await freeze(mono, 0.5, FAST);
    assert.ok(peak(out) > 0);
    // Both channels draw their own phases, so a mono source still decorrelates
    // into a stereo pad rather than collapsing to the centre.
    assert.ok(maxAbsDiff(out.l, out.r) > 1e-3);
  });
});

// ── Determinism ─────────────────────────────────────────────

describe('freeze — determinism', () => {
  it('is bit-identical for the same seed', async () => {
    const src = sweepSource();
    const a = await freeze(src, 0.5, { ...FAST, seed: 12345n });
    const b = await freeze(src, 0.5, { ...FAST, seed: 12345n });
    assert.deepEqual(Array.from(a.l), Array.from(b.l));
    assert.deepEqual(Array.from(a.r), Array.from(b.r));
  });

  it('produces different audio for different seeds', async () => {
    const src = sweepSource();
    const a = await freeze(src, 0.5, { ...FAST, seed: 1n });
    const b = await freeze(src, 0.5, { ...FAST, seed: 2n });
    assert.ok(maxAbsDiff(a.l, b.l) > 1e-3);
  });

  it('defaults to DEFAULT_SEED', async () => {
    const src = sweepSource();
    const implicit = await freeze(src, 0.5, FAST);
    const explicit = await freeze(src, 0.5, { ...FAST, seed: DEFAULT_SEED });
    assert.deepEqual(Array.from(implicit.l), Array.from(explicit.l));
  });

  it('is deterministic in scan mode too', async () => {
    const src = sweepSource();
    const a = await freeze(src, 0.5, { ...FAST, scan: 0.8, seed: 7n });
    const b = await freeze(src, 0.5, { ...FAST, scan: 0.8, seed: 7n });
    assert.deepEqual(Array.from(a.l), Array.from(b.l));
  });
});

// ── Normalisation ───────────────────────────────────────────

describe('freeze — normalisation', () => {
  it('peak-normalises to 0.92', async () => {
    const out = await freeze(sweepSource(), 0.5, FAST);
    assert.ok(Math.abs(peak(out) - 0.92) < 1e-6);
  });

  it('normalises to 0.92 regardless of source level', async () => {
    const quiet = sweepSource();
    for (let i = 0; i < quiet.l.length; i++) { quiet.l[i] *= 1e-4; quiet.r[i] *= 1e-4; }
    const out = await freeze(quiet, 0.5, FAST);
    assert.ok(Math.abs(peak(out) - 0.92) < 1e-6);
  });

  it('scales both channels by one gain, preserving the stereo image', async () => {
    // The source is 0.7× on the right, so the frozen pad's channel balance
    // must survive normalisation rather than being levelled per channel.
    const out = await freeze(sweepSource(), 1, FAST);
    let sumL = 0;
    let sumR = 0;
    for (let i = 0; i < out.l.length; i++) { sumL += out.l[i] * out.l[i]; sumR += out.r[i] * out.r[i]; }
    const ratio = Math.sqrt(sumR / sumL);
    assert.ok(ratio > 0.6 && ratio < 0.8, `channel ratio ${ratio} should stay near 0.7`);
  });
});

// ── Parameters ──────────────────────────────────────────────

describe('freeze — parameters', () => {
  it('clamps positionNorm below 0 and above 1', async () => {
    const src = sweepSource();
    const low = await freeze(src, 0.5, { ...FAST, positionNorm: -3 });
    const zero = await freeze(src, 0.5, { ...FAST, positionNorm: 0 });
    const high = await freeze(src, 0.5, { ...FAST, positionNorm: 42 });
    const one = await freeze(src, 0.5, { ...FAST, positionNorm: 1 });
    assert.deepEqual(Array.from(low.l), Array.from(zero.l));
    assert.deepEqual(Array.from(high.l), Array.from(one.l));
  });

  it('captures a different spectrum at a different position', async () => {
    const src = sweepSource();
    const early = await freeze(src, 0.5, { ...FAST, positionNorm: 0 });
    const late = await freeze(src, 0.5, { ...FAST, positionNorm: 1 });
    assert.ok(maxAbsDiff(early.l, late.l) > 1e-3);
  });

  it('skips the blur at or below smear 0.01', async () => {
    // The kernel tests `smear > 0.01`, so 0.01 exactly is still a clean freeze.
    const src = sweepSource();
    const none = await freeze(src, 0.5, FAST);
    const barely = await freeze(src, 0.5, { ...FAST, smear: 0.01 });
    assert.deepEqual(Array.from(none.l), Array.from(barely.l));
  });

  it('smearing changes the spectrum', async () => {
    const src = sweepSource();
    const none = await freeze(src, 0.5, FAST);
    const smeared = await freeze(src, 0.5, { ...FAST, smear: 0.6 });
    assert.ok(maxAbsDiff(none.l, smeared.l) > 1e-3);
  });

  it('clamps smear above 1', async () => {
    const src = sweepSource();
    const full = await freeze(src, 0.5, { ...FAST, smear: 1 });
    const over = await freeze(src, 0.5, { ...FAST, smear: 9 });
    assert.deepEqual(Array.from(full.l), Array.from(over.l));
  });

  it('scanning drifts the capture point, so the output evolves', async () => {
    const src = sweepSource();
    const still = await freeze(src, 1, { ...FAST, positionNorm: 0 });
    const scanned = await freeze(src, 1, { ...FAST, positionNorm: 0, scan: 1 });
    assert.ok(maxAbsDiff(still.l, scanned.l) > 1e-3);
    // Early hops sit near the capture point of the static freeze; late hops
    // have scanned away from it, so the divergence grows over the render.
    const n = still.l.length;
    const head = maxAbsDiff(still.l.subarray(0, 256), scanned.l.subarray(0, 256));
    const tail = maxAbsDiff(still.l.subarray(n - 256), scanned.l.subarray(n - 256));
    assert.ok(tail > head, `tail divergence ${tail} should exceed head ${head}`);
  });

  it('clamps scan above 1', async () => {
    const src = sweepSource();
    const full = await freeze(src, 0.5, { ...FAST, scan: 1 });
    const over = await freeze(src, 0.5, { ...FAST, scan: 5 });
    assert.deepEqual(Array.from(full.l), Array.from(over.l));
  });

  it('a negative scan is treated as a static freeze', async () => {
    const src = sweepSource();
    const still = await freeze(src, 0.5, { ...FAST, scan: 0 });
    const negative = await freeze(src, 0.5, { ...FAST, scan: -1 });
    assert.deepEqual(Array.from(still.l), Array.from(negative.l));
  });

  it('a longer window changes the freeze', async () => {
    const src = sweepSource();
    const short = await freeze(src, 0.5, { windowSec: 0.05 });
    const long = await freeze(src, 0.5, { windowSec: 0.1 });
    assert.equal(short.l.length, long.l.length);
    assert.ok(maxAbsDiff(short.l, long.l) > 1e-3);
  });
});

// ── Progress hook ───────────────────────────────────────────

describe('freeze — progress', () => {
  it('reports fractions that climb to 1', async () => {
    const seen = [];
    await freeze(sweepSource(), 1, { ...FAST, onProgress: (f) => { seen.push(f); } });
    assert.ok(seen.length >= 2);
    assert.equal(seen[0], 0);
    assert.equal(seen.at(-1), 1);
    for (let i = 0; i < seen.length; i++) {
      assert.ok(seen[i] >= 0 && seen[i] <= 1, `fraction ${seen[i]} out of range`);
      if (i > 0) assert.ok(seen[i] >= seen[i - 1], 'fractions must not go backwards');
    }
  });

  it('reports every 32nd hop', async () => {
    // 1 s at 8 kHz with a 512-frame window is 8000 frames on a 128-frame hop:
    // hops 0..60, so hops 0/32 plus the final 1.
    const seen = [];
    await freeze(sweepSource(), 1, { ...FAST, onProgress: (f) => { seen.push(f); } });
    assert.equal(seen.length, 3);
  });

  it('awaits a promise-returning hook', async () => {
    const order = [];
    await freeze(sweepSource(), 1, {
      ...FAST,
      onProgress: async (f) => {
        order.push(`start:${f}`);
        await new Promise((r) => setImmediate(r));
        order.push(`end:${f}`);
      },
    });
    // Nothing overlaps: each callback finishes before the next one starts.
    for (let i = 0; i < order.length; i += 2) {
      assert.ok(order[i].startsWith('start:'));
      assert.ok(order[i + 1].startsWith('end:'));
      assert.equal(order[i].slice(6), order[i + 1].slice(4));
    }
  });

  it('produces identical audio with and without a hook', async () => {
    const src = sweepSource();
    const bare = await freeze(src, 0.5, FAST);
    const hooked = await freeze(src, 0.5, { ...FAST, onProgress: () => {} });
    assert.deepEqual(Array.from(bare.l), Array.from(hooked.l));
  });

  it('works with no options at all', async () => {
    const out = await freeze(sweepSource(2), 0.5);
    assert.ok(out.l.length > 0);
    assert.ok(isFiniteChannel(out.l));
  });
});
