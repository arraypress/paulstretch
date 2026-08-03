/**
 * Granular cloud: parity with `GranularKernel.swift` (swift-paul-stretch), plus
 * the edge cases the JS port has to get right on its own.
 *
 * The Swift reference is rendered with fadeIn/fadeOut = 0: `StretchRenderer`
 * applies a 20 s fade envelope to one-shot renders by default, which is a
 * renderer stage, not part of the granular engine — comparing with them on
 * would be comparing envelopes rather than engines.
 *
 * Fingerprints are RMS of 32 equal slices (ceil(n/32) chunking, matching the
 * Swift harness) of the reference render for a fixed source and seed, with both
 * sides peak-normalised to unity first — see the note in `fingerprint()`.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { granular } from '../src/granular.js';
import { fromChannels, createBuffer, peak, duration } from '../src/buffer.js';
import { DEFAULT_SEED } from '../src/stretch.js';

const SWIFT = {
  base: [0.068605757, 0.218087196, 0.254571959, 0.144546169, 0.294500826, 0.273261402, 0.346222275, 0.249829293, 0.185405537, 0.205005922, 0.194311134, 0.270435012, 0.269505308, 0.240166275, 0.421627165, 0.191365258, 0.189092286, 0.213620049, 0.243891181, 0.242031505, 0.241138486, 0.262457249, 0.202773409, 0.138678189, 0.225341290, 0.283652120, 0.164456991, 0.157922059, 0.230659826, 0.191763978, 0.183212475, 0.206603068],
  dense: [0.117975821, 0.234611765, 0.232774172, 0.217765598, 0.217139121, 0.263081048, 0.275419864, 0.240067550, 0.277243044, 0.235026157, 0.331679737, 0.238612753, 0.204017613, 0.298200205, 0.260337653, 0.275370502, 0.231515869, 0.234278672, 0.196277806, 0.268799094, 0.240923167, 0.236570754, 0.259323825, 0.233132588, 0.225363595, 0.281686252, 0.251231635, 0.230548289, 0.230241349, 0.198564052, 0.195687223, 0.153445562],
  rigid: [0.000000000, 0.384416215, 0.532681108, 0.493442895, 0.474540265, 0.368106586, 0.424328147, 0.478253478, 0.534416920, 0.474987284, 0.474930237, 0.412653035, 0.474686244, 0.520537093, 0.489025573, 0.420220295, 0.422803291, 0.422691395, 0.406556501, 0.366812235, 0.505742920, 0.524857314, 0.521393138, 0.570576786, 0.564995322, 0.534682214, 0.466562844, 0.442068964, 0.433895873, 0.447034012, 0.508928340, 0.562503372],
};

// name, grainSeconds, density, positionJitter, timeJitter, pitchSpread
const CASES = [
  ['base', 0.15, 8, 0.05, 0.75, 0],
  ['dense', 0.08, 16, 0.20, 0.50, 3],
  ['rigid', 0.25, 4, 0.00, 0.00, 0],
];
const TOL = 1e-5;

/** The fixed three-partial source the Swift fingerprints were captured from. */
function makeSource(seconds = 2, sr = 44100) {
  const n = sr * seconds;
  const l = new Float32Array(n);
  const r = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const t = i / sr;
    const v = 0.5 * Math.sin(2 * Math.PI * 220 * t) + 0.3 * Math.sin(2 * Math.PI * 437 * t) + 0.2 * Math.sin(2 * Math.PI * 1103 * t);
    l[i] = v;
    r[i] = v * 0.8;
  }
  return fromChannels(l, r, sr);
}

/**
 * RMS of 32 slices of a channel, peak-normalised to unity first.
 *
 * Swift's granular output gain comes from a SWEEP-approximated peak
 * (ChunkedRenderer measures peaks at ~12 s granularity for granular and
 * phase-vocoder, leaving deliberate headroom), so its absolute level is not
 * reproducible by an exact peak normalise — and shouldn't be. Both sides are
 * normalised to unity here so the test checks the audio, which matches
 * sample-for-sample.
 */
function fingerprint(raw) {
  let pk = 0;
  for (let i = 0; i < raw.length; i++) pk = Math.max(pk, Math.abs(raw[i]));
  const ch = new Float32Array(raw.length);
  for (let i = 0; i < raw.length; i++) ch[i] = raw[i] / (pk || 1);

  const size = Math.ceil(ch.length / 32);
  const out = [];
  for (let s = 0; s < 32; s++) {
    const slice = ch.subarray(s * size, Math.min(ch.length, (s + 1) * size));
    let acc = 0;
    for (let i = 0; i < slice.length; i++) acc += slice[i] * slice[i];
    out.push(Math.sqrt(acc / slice.length));
  }
  return out;
}

describe('granular — Swift parity', () => {
  for (const [name, grainSeconds, density, positionJitter, timeJitter, pitchSpread] of CASES) {
    it(`matches the Swift reference: ${name}`, async () => {
      const input = makeSource();
      const out = await granular(input, 4, {
        grainSeconds, density, positionJitter, timeJitter, pitchSpread, panSpread: 0.6,
      });
      const got = fingerprint(out.l);
      const want = SWIFT[name];
      let worst = 0;
      for (let s = 0; s < 32; s++) worst = Math.max(worst, Math.abs(got[s] - want[s]));
      assert.ok(worst < TOL, `${name} diverges from Swift: ${worst.toExponential(2)} > ${TOL}`);
    });
  }
});

describe('granular — output shape', () => {
  it('renders the requested duration', async () => {
    const out = await granular(makeSource(1), 3);
    assert.equal(out.l.length, 3 * 44100);
    assert.equal(out.r.length, 3 * 44100);
    assert.equal(out.sampleRate, 44100);
    assert.equal(duration(out), 3);
  });

  it('truncates a fractional frame count rather than rounding', async () => {
    // 0.5000001 s at 44100 is 22050.004… frames; Swift's Int() truncates.
    const out = await granular(makeSource(1), 0.5000001, { grainSeconds: 0.01 });
    assert.equal(out.l.length, 22050);
  });

  it('always produces at least one frame', async () => {
    for (const seconds of [0, -1, 1e-9]) {
      const out = await granular(makeSource(1), seconds, { grainSeconds: 0.01 });
      assert.equal(out.l.length, 1, `targetSeconds ${seconds}`);
    }
  });

  it('gives the two channels independent storage', async () => {
    const out = await granular(makeSource(1), 1, { panSpread: 1 });
    assert.notEqual(out.l, out.r);
  });

  it('normalises the louder channel to 0.92', async () => {
    const out = await granular(makeSource(1), 2, { panSpread: 0.6 });
    assert.ok(Math.abs(peak(out) - 0.92) < 1e-6);
  });
});

describe('granular — degenerate inputs', () => {
  it('returns silence for an empty source', async () => {
    const out = await granular(createBuffer(0, 44100), 1);
    assert.equal(out.l.length, 44100);
    assert.equal(peak(out), 0);
  });

  it('reports completion for an empty source', async () => {
    const seen = [];
    await granular(createBuffer(0, 44100), 1, { onProgress: (f) => seen.push(f) });
    assert.deepEqual(seen, [1]);
  });

  it('handles a one-frame source without reading out of bounds', async () => {
    const one = fromChannels(new Float32Array([0.5]), undefined, 44100);
    const out = await granular(one, 0.5, { grainSeconds: 0.01, positionJitter: 0.5 });
    assert.equal(out.l.length, 22050);
    for (let i = 0; i < out.l.length; i++) {
      assert.ok(Number.isFinite(out.l[i]) && Number.isFinite(out.r[i]), `NaN at ${i}`);
    }
  });

  it('produces silence from a silent source rather than dividing by zero', async () => {
    const out = await granular(createBuffer(44100, 44100), 1);
    assert.equal(peak(out), 0);
  });

  it('accepts a mono source whose channels share one array', async () => {
    const l = new Float32Array(44100);
    for (let i = 0; i < l.length; i++) l[i] = Math.sin((2 * Math.PI * 220 * i) / 44100);
    const before = Float32Array.from(l);
    const mono = fromChannels(l, undefined, 44100);
    const out = await granular(mono, 1, { panSpread: 0 });
    assert.ok(peak(out) > 0);
    // Centre pan is equal-power, so an untouched mono source stays balanced.
    for (let i = 0; i < out.l.length; i++) {
      assert.ok(Math.abs(out.l[i] - out.r[i]) < 1e-6, `channels diverge at ${i}`);
    }
    assert.deepEqual(Array.from(l), Array.from(before), 'source must not be mutated');
  });
});

describe('granular — parameter clamping', () => {
  const input = makeSource(1);

  it('treats a negative pitchSpread as zero', async () => {
    const a = await granular(input, 1, { pitchSpread: -5 });
    const b = await granular(input, 1, { pitchSpread: 0 });
    assert.deepEqual(Array.from(a.l), Array.from(b.l));
  });

  it('clamps panSpread above 1 and below 0', async () => {
    const wide = await granular(input, 1, { panSpread: 5 });
    const hard = await granular(input, 1, { panSpread: 1 });
    assert.deepEqual(Array.from(wide.l), Array.from(hard.l));

    const under = await granular(input, 1, { panSpread: -2 });
    const centred = await granular(input, 1, { panSpread: 0 });
    assert.deepEqual(Array.from(under.l), Array.from(centred.l));
  });

  it('clamps positionJitter and timeJitter to 0..1', async () => {
    const over = await granular(input, 1, { positionJitter: 3, timeJitter: 3 });
    const full = await granular(input, 1, { positionJitter: 1, timeJitter: 1 });
    assert.deepEqual(Array.from(over.l), Array.from(full.l));
  });

  it('treats a density below 1 as 1', async () => {
    const sparse = await granular(input, 1, { density: 0.1 });
    const one = await granular(input, 1, { density: 1 });
    assert.deepEqual(Array.from(sparse.l), Array.from(one.l));
  });

  it('floors grainSeconds at 5 ms', async () => {
    const tiny = await granular(input, 1, { grainSeconds: 0.0001 });
    const floored = await granular(input, 1, { grainSeconds: 0.005 });
    assert.deepEqual(Array.from(tiny.l), Array.from(floored.l));
  });

  it('floors the grain length at 64 frames for very low sample rates', async () => {
    // 0.005 s at 1000 Hz is 5 frames; the kernel must still build a usable
    // Hann window, so it clamps to 64.
    const slow = fromChannels(new Float32Array(2000).fill(0.5), undefined, 1000);
    const out = await granular(slow, 1, { grainSeconds: 0.005 });
    assert.equal(out.l.length, 1000);
    assert.ok(peak(out) > 0);
  });
});

describe('granular — determinism', () => {
  const input = makeSource(1);

  it('is bit-identical across renders with the same seed', async () => {
    const opts = { pitchSpread: 4, panSpread: 0.8, positionJitter: 0.3 };
    const a = await granular(input, 2, opts);
    const b = await granular(input, 2, opts);
    assert.deepEqual(Array.from(a.l), Array.from(b.l));
    assert.deepEqual(Array.from(a.r), Array.from(b.r));
  });

  it('defaults to DEFAULT_SEED', async () => {
    const implicit = await granular(input, 1, { panSpread: 0.5 });
    const explicit = await granular(input, 1, { panSpread: 0.5, seed: DEFAULT_SEED });
    assert.deepEqual(Array.from(implicit.l), Array.from(explicit.l));
  });

  it('produces different audio for a different seed', async () => {
    const a = await granular(input, 1, { seed: 1n, panSpread: 0.5 });
    const b = await granular(input, 1, { seed: 2n, panSpread: 0.5 });
    let diff = 0;
    for (let i = 0; i < a.l.length; i++) diff = Math.max(diff, Math.abs(a.l[i] - b.l[i]));
    assert.ok(diff > 1e-3, `seeds 1 and 2 gave near-identical output (${diff})`);
  });

  it('ignores the seed when every random range is zero', async () => {
    // With no jitter and no spread the draws still happen (fixed draw order),
    // but nothing consumes them, so the render is seed-independent.
    const rigid = { grainSeconds: 0.25, density: 4, positionJitter: 0, timeJitter: 0, pitchSpread: 0, panSpread: 0 };
    const a = await granular(input, 1, { ...rigid, seed: 11n });
    const b = await granular(input, 1, { ...rigid, seed: 99n });
    assert.deepEqual(Array.from(a.l), Array.from(b.l));
  });
});

describe('granular — progress', () => {
  it('reports monotonic progress ending at 1', async () => {
    const seen = [];
    await granular(makeSource(1), 4, { grainSeconds: 0.08, density: 16, onProgress: (f) => { seen.push(f); } });
    assert.ok(seen.length > 1, 'expected in-flight progress for an 800-grain render');
    for (let i = 1; i < seen.length; i++) assert.ok(seen[i] >= seen[i - 1], 'progress went backwards');
    assert.ok(seen[0] > 0 && seen[0] <= 1);
    assert.equal(seen[seen.length - 1], 1);
  });

  it('awaits a promise-returning hook', async () => {
    let resolved = 0;
    let calls = 0;
    await granular(makeSource(1), 4, {
      grainSeconds: 0.08,
      density: 16,
      onProgress: async () => {
        calls++;
        await Promise.resolve();
        resolved++;
      },
    });
    assert.ok(calls > 1);
    assert.equal(resolved, calls, 'every hook promise should have settled before returning');
  });

  it('produces identical audio with and without the hook', async () => {
    const input = makeSource(1);
    const withHook = await granular(input, 2, { grainSeconds: 0.08, density: 16, onProgress: () => {} });
    const without = await granular(input, 2, { grainSeconds: 0.08, density: 16 });
    assert.deepEqual(Array.from(withHook.l), Array.from(without.l));
  });
});
