/**
 * Phase-vocoder tests.
 *
 * The `Swift parity` block is the important one: the fingerprints below are
 * RMS profiles captured from real `PhaseVocoder.swift` renders
 * (swift-paul-stretch), so they are what makes this port trustworthy rather
 * than merely self-consistent. The reference was rendered with fades disabled
 * (StretchRenderer applies a 20 s envelope to one-shot renders — a renderer
 * stage, not the engine), and both sides peak-normalised to unity since Swift
 * derives the phase-vocoder output gain from a sweep-approximated peak.
 *
 * TOLERANCE: measured null depth is -56 dB (correlation 0.999999), NOT the
 * -130 dB the other engines reach. That is inherent, not a defect: the phase
 * vocoder ACCUMULATES phase across ~1400 windows, so the sub-float32
 * disagreement between Accelerate's vDSP FFT and our radix-2 compounds. Every
 * other engine derives each window independently from a seed and so never
 * accumulates. 1e-3 is set to catch real algorithmic drift while tolerating
 * that; do not tighten it without changing the FFT.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { phaseVocoder } from '../src/phase-vocoder.js';
import { createBuffer, fromChannels, peak } from '../src/buffer.js';

const SWIFT = {
  x4: [0.243301497, 0.416309832, 0.416328834, 0.416923624, 0.416922469, 0.416329757, 0.416328938, 0.416923486, 0.416922454, 0.416329938, 0.416328852, 0.416923439, 0.416922596, 0.416329897, 0.416328734, 0.416923554, 0.416922634, 0.416329744, 0.416328797, 0.416923660, 0.416922510, 0.416329749, 0.416328930, 0.416923542, 0.416922467, 0.416329918, 0.416584686, 0.424617483, 0.376620838, 0.090744325, 0.172686572, 0.035111574],
  x2: [0.050208346, 0.268366301, 0.330441181, 0.330224191, 0.330278850, 0.330421573, 0.330661603, 0.330984757, 0.331099931, 0.330546364, 0.330560272, 0.330130150, 0.330303227, 0.330391528, 0.330700591, 0.331016157, 0.331098310, 0.330522614, 0.330612538, 0.330055331, 0.330282029, 0.330391237, 0.330703437, 0.331173707, 0.330994354, 0.330571518, 0.330592978, 0.330914573, 0.333821040, 0.330524982, 0.089882227, 0.068795753],
  pitch: [0.331302589, 0.385210029, 0.391112482, 0.394788099, 0.386313771, 0.388939132, 0.395505276, 0.388342240, 0.386807065, 0.394790492, 0.390500875, 0.385343809, 0.393570591, 0.392650968, 0.385559196, 0.391641424, 0.394263187, 0.386698373, 0.389043270, 0.395187251, 0.388062051, 0.386797479, 0.394860625, 0.390467948, 0.385840566, 0.393308307, 0.392455963, 0.384776107, 0.379371519, 0.146689160, 0.203616331, 0.063020387],
};

// name, ratio, pitchSemitones  (source is 2 s, so ratio = target / 2)
const CASES = [
  ['x4', 4, 0],
  ['x2', 2, 0],
  ['pitch', 4, 5],
];

const TOL = 1e-3;

/** The exact source the Swift fingerprints were captured from: 2 s of three
 *  inharmonic sines, right channel 0.8x the left. */
function referenceInput() {
  const sr = 44100;
  const n = sr * 2;
  const L = new Float32Array(n);
  const R = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const t = i / sr;
    const v = 0.5 * Math.sin(2 * Math.PI * 220 * t) + 0.3 * Math.sin(2 * Math.PI * 437 * t) + 0.2 * Math.sin(2 * Math.PI * 1103 * t);
    L[i] = v;
    R[i] = v * 0.8;
  }
  return fromChannels(L, R, sr);
}

/** Worst per-slice deviation between a channel's 32-slice RMS profile and a
 *  captured fingerprint, after normalising the channel to unity peak. */
function fingerprintError(raw, want) {
  let pk = 0;
  for (let i = 0; i < raw.length; i++) pk = Math.max(pk, Math.abs(raw[i]));
  const size = Math.ceil(raw.length / 32);
  let worst = 0;
  for (let s = 0; s < 32; s++) {
    const sl = raw.subarray(s * size, Math.min(raw.length, (s + 1) * size));
    let acc = 0;
    for (let i = 0; i < sl.length; i++) { const v = sl[i] / (pk || 1); acc += v * v; }
    worst = Math.max(worst, Math.abs(Math.sqrt(acc / sl.length) - want[s]));
  }
  return worst;
}

/** A short, cheap source for the behavioural tests. */
function toneInput(frames = 8000, sr = 44100) {
  const L = new Float32Array(frames);
  const R = new Float32Array(frames);
  for (let i = 0; i < frames; i++) {
    const t = i / sr;
    L[i] = 0.6 * Math.sin(2 * Math.PI * 330 * t);
    R[i] = 0.4 * Math.sin(2 * Math.PI * 495 * t);
  }
  return fromChannels(L, R, sr);
}

function rms(a, from = 0, to = a.length) {
  let acc = 0;
  for (let i = from; i < to; i++) acc += a[i] * a[i];
  return Math.sqrt(acc / Math.max(1, to - from));
}

function allFinite(buffer) {
  for (let i = 0; i < buffer.l.length; i++) {
    if (!Number.isFinite(buffer.l[i]) || !Number.isFinite(buffer.r[i])) return false;
  }
  return true;
}

describe('phaseVocoder — Swift parity', () => {
  for (const [name, ratio, semis] of CASES) {
    it(`${name} matches the Swift fingerprint`, async () => {
      const out = await phaseVocoder(referenceInput(), ratio, {
        windowSec: 0.25,
        pitchSemitones: semis,
      });
      const worst = fingerprintError(out.l, SWIFT[name]);
      assert.ok(worst < TOL, `pv/${name} diverges (${worst.toExponential(2)} > ${TOL})`);
    });
  }
});

describe('phaseVocoder — output geometry', () => {
  it('natural length is max(window, trunc(inputFrames * ratio))', async () => {
    const input = toneInput(8000);
    const out = await phaseVocoder(input, 4, { windowSec: 0.02 });
    // 0.02 s at 44100 -> nextPow2(882) = 1024 frames, so the stretch wins.
    assert.equal(out.l.length, 32000);
    assert.equal(out.r.length, 32000);
    assert.equal(out.sampleRate, 44100);
  });

  it('never returns fewer frames than one window', async () => {
    // 8000 frames compressed 10x is 800 frames, below the 1024-frame window.
    const out = await phaseVocoder(toneInput(8000), 0.1, { windowSec: 0.02 });
    assert.equal(out.l.length, 1024);
  });

  it('outputFrames delivers exactly that many, stopping short', async () => {
    const out = await phaseVocoder(toneInput(8000), 4, { windowSec: 0.02, outputFrames: 5000 });
    assert.equal(out.l.length, 5000);
    assert.ok(rms(out.l) > 0, 'the delivered range is not silent');
  });

  it('outputFrames past the last window zero-pads', async () => {
    // The window schedule is driven by the natural output length, but the
    // final window still spans a whole W from its own start — so writing
    // stops at lastBlock*hop + W, a little past `natural`, and everything
    // after that is untouched allocation.
    const W = 1024;
    const hop = W >> 2;
    const natural = 32000;
    const lastWritten = Math.trunc((natural - (W >> 1) - 1) / hop) * hop + W;
    const out = await phaseVocoder(toneInput(8000), 4, { windowSec: 0.02, outputFrames: natural + 4000 });
    assert.equal(out.l.length, natural + 4000);
    for (let i = lastWritten; i < out.l.length; i++) {
      assert.equal(out.l[i], 0);
      assert.equal(out.r[i], 0);
    }
  });

  it('truncating outputFrames matches the head of the untruncated render', async () => {
    const input = toneInput(8000);
    const full = await phaseVocoder(input, 4, { windowSec: 0.02 });
    const cut = await phaseVocoder(input, 4, { windowSec: 0.02, outputFrames: 5000 });
    for (let i = 0; i < 5000; i++) assert.equal(cut.l[i], full.l[i]);
  });

  it('clamps a zero or negative outputFrames to one frame', async () => {
    const zero = await phaseVocoder(toneInput(2000), 2, { windowSec: 0.01, outputFrames: 0 });
    const neg = await phaseVocoder(toneInput(2000), 2, { windowSec: 0.01, outputFrames: -100 });
    assert.equal(zero.l.length, 1);
    assert.equal(neg.l.length, 1);
  });

  it('rounds a fractional outputFrames toward zero', async () => {
    const out = await phaseVocoder(toneInput(2000), 2, { windowSec: 0.01, outputFrames: 999.9 });
    assert.equal(out.l.length, 999);
  });
});

describe('phaseVocoder — empty and tiny inputs', () => {
  it('an empty input yields a silent window-length buffer', async () => {
    const out = await phaseVocoder(createBuffer(0, 44100), 4, { windowSec: 0.02 });
    assert.equal(out.l.length, 1024);
    assert.equal(peak(out), 0);
  });

  it('an empty input still honours outputFrames', async () => {
    const out = await phaseVocoder(createBuffer(0, 44100), 4, { windowSec: 0.02, outputFrames: 700 });
    assert.equal(out.l.length, 700);
    assert.equal(peak(out), 0);
  });

  it('an empty input still reports completion', async () => {
    const seen = [];
    await phaseVocoder(createBuffer(0, 44100), 4, { windowSec: 0.02, onProgress: (f) => { seen.push(f); } });
    assert.deepEqual(seen, [1]);
  });

  it('an empty input with wrapInput does not divide by zero', async () => {
    const out = await phaseVocoder(createBuffer(0, 44100), 4, { windowSec: 0.02, wrapInput: true });
    assert.equal(peak(out), 0);
    assert.ok(allFinite(out));
  });

  it('a single-frame input produces finite audio', async () => {
    const one = fromChannels(new Float32Array([0.5]), new Float32Array([-0.5]), 44100);
    const out = await phaseVocoder(one, 4, { windowSec: 0.02 });
    assert.equal(out.l.length, 1024);
    assert.ok(allFinite(out));
  });

  it('an input shorter than one window produces finite audio', async () => {
    const out = await phaseVocoder(toneInput(300), 8, { windowSec: 0.02 });
    assert.ok(allFinite(out));
    assert.ok(peak(out) > 0, 'the short source still reaches the output');
  });
});

describe('phaseVocoder — determinism', () => {
  // There is no seed to vary: this engine propagates phase rather than
  // randomising it, so nothing in the render is random. The property worth
  // pinning is that two runs agree exactly, and that the parameters that DO
  // exist actually change the result.
  it('two renders of the same input are bit-identical', async () => {
    const input = toneInput(8000);
    const a = await phaseVocoder(input, 4, { windowSec: 0.02, pitchSemitones: 3 });
    const b = await phaseVocoder(input, 4, { windowSec: 0.02, pitchSemitones: 3 });
    assert.deepEqual(Array.from(a.l), Array.from(b.l));
    assert.deepEqual(Array.from(a.r), Array.from(b.r));
  });

  it('does not mutate its input', async () => {
    const input = toneInput(4000);
    const before = Float32Array.from(input.l);
    await phaseVocoder(input, 4, { windowSec: 0.02 });
    assert.deepEqual(Array.from(input.l), Array.from(before));
  });

  it('a different ratio gives a different render', async () => {
    const input = toneInput(8000);
    const a = await phaseVocoder(input, 4, { windowSec: 0.02, outputFrames: 8000 });
    const b = await phaseVocoder(input, 6, { windowSec: 0.02, outputFrames: 8000 });
    assert.notDeepEqual(Array.from(a.l), Array.from(b.l));
  });

  it('a different window size gives a different render', async () => {
    const input = toneInput(8000);
    const a = await phaseVocoder(input, 4, { windowSec: 0.02, outputFrames: 8000 });
    const b = await phaseVocoder(input, 4, { windowSec: 0.05, outputFrames: 8000 });
    assert.notDeepEqual(Array.from(a.l), Array.from(b.l));
  });
});

describe('phaseVocoder — pitch shifting', () => {
  it('shifts the spectrum when asked', async () => {
    const input = toneInput(8000);
    const flat = await phaseVocoder(input, 4, { windowSec: 0.05, outputFrames: 16000 });
    const up = await phaseVocoder(input, 4, { windowSec: 0.05, pitchSemitones: 7, outputFrames: 16000 });
    assert.notDeepEqual(Array.from(flat.l), Array.from(up.l));
    assert.ok(allFinite(up));
  });

  it('treats a sub-threshold shift as no shift at all', async () => {
    // |2^(s/12) - 1| <= 0.001 skips the bin remap entirely, so 0.01 semitones
    // must be bit-identical to 0.
    const input = toneInput(4000);
    const none = await phaseVocoder(input, 4, { windowSec: 0.02 });
    const tiny = await phaseVocoder(input, 4, { windowSec: 0.02, pitchSemitones: 0.01 });
    assert.deepEqual(Array.from(tiny.l), Array.from(none.l));
  });

  it('applies a shift just past the threshold', async () => {
    const input = toneInput(4000);
    const none = await phaseVocoder(input, 4, { windowSec: 0.02 });
    const just = await phaseVocoder(input, 4, { windowSec: 0.02, pitchSemitones: 0.05 });
    assert.notDeepEqual(Array.from(just.l), Array.from(none.l));
  });

  it('shifts down as well as up', async () => {
    const input = toneInput(8000);
    const down = await phaseVocoder(input, 4, { windowSec: 0.05, pitchSemitones: -12, outputFrames: 16000 });
    assert.ok(allFinite(down));
    assert.ok(peak(down) > 0);
  });

  it('an extreme upward shift empties the spectrum rather than ringing', async () => {
    // Every output bin sources from below bin 1, so the remap zeroes them all.
    const out = await phaseVocoder(toneInput(4000), 2, { windowSec: 0.02, pitchSemitones: -200 });
    assert.ok(allFinite(out));
  });
});

describe('phaseVocoder — wrapInput', () => {
  it('keeps the tail alive where zero-padding fades out', async () => {
    const input = toneInput(8000);
    const padded = await phaseVocoder(input, 4, { windowSec: 0.05 });
    const wrapped = await phaseVocoder(input, 4, { windowSec: 0.05, wrapInput: true });
    assert.equal(padded.l.length, wrapped.l.length);
    const n = padded.l.length;
    const tailPadded = rms(padded.l, n - 1024, n);
    const tailWrapped = rms(wrapped.l, n - 1024, n);
    assert.ok(tailWrapped > tailPadded, `wrapped tail ${tailWrapped} should exceed padded ${tailPadded}`);
  });

  it('agrees with zero-padding over the range neither reads past', async () => {
    const input = toneInput(8000);
    const padded = await phaseVocoder(input, 4, { windowSec: 0.02 });
    const wrapped = await phaseVocoder(input, 4, { windowSec: 0.02, wrapInput: true });
    for (let i = 0; i < 4000; i++) assert.equal(wrapped.l[i], padded.l[i]);
  });
});

describe('phaseVocoder — level', () => {
  it('does NOT normalise: raw overlap-add runs hot', async () => {
    // The documented contract. A ~0.6 peak source comes back past full scale
    // because 4x-Hann overlap-add sums to roughly 1.5x, and callers are
    // expected to run normalizeToPeak themselves.
    const out = await phaseVocoder(toneInput(8000), 4, { windowSec: 0.05 });
    assert.ok(peak(out) > 0.6, `expected a hot output, got ${peak(out)}`);
  });

  it('compresses as happily as it stretches', async () => {
    const out = await phaseVocoder(toneInput(16000), 0.5, { windowSec: 0.02 });
    assert.equal(out.l.length, 8000);
    assert.ok(peak(out) > 0);
    assert.ok(allFinite(out));
  });

  it('a ratio of 1 passes the material through recognisably', async () => {
    const input = toneInput(8000);
    const out = await phaseVocoder(input, 1, { windowSec: 0.02 });
    assert.equal(out.l.length, 8000);
    // Steady tones survive a unity pass; compare interior RMS, ignoring the
    // window-length ramps at either end.
    const ratio = rms(out.l, 2048, 6000) / rms(input.l, 2048, 6000);
    assert.ok(ratio > 0.8 && ratio < 2.5, `unity-ratio level ratio was ${ratio}`);
  });
});

describe('phaseVocoder — channels', () => {
  it('a mono buffer comes back with identical channels', async () => {
    const mono = new Float32Array(8000);
    for (let i = 0; i < mono.length; i++) mono[i] = 0.5 * Math.sin((2 * Math.PI * 440 * i) / 44100);
    const out = await phaseVocoder(fromChannels(mono, undefined, 44100), 4, { windowSec: 0.02 });
    assert.notEqual(out.l, out.r, 'the output owns two independent arrays');
    assert.deepEqual(Array.from(out.l), Array.from(out.r));
  });

  it('keeps genuinely different channels different', async () => {
    const out = await phaseVocoder(toneInput(8000), 4, { windowSec: 0.02 });
    assert.notDeepEqual(Array.from(out.l), Array.from(out.r));
  });

  it('carries the input sample rate through', async () => {
    const out = await phaseVocoder(toneInput(8000, 48000), 4, { windowSec: 0.02 });
    assert.equal(out.sampleRate, 48000);
  });
});

describe('phaseVocoder — onProgress', () => {
  it('is never called when omitted', async () => {
    // Nothing to assert directly; the point is that the render completes
    // without touching a hook, so this just pins the no-hook path.
    const out = await phaseVocoder(toneInput(8000), 4, { windowSec: 0.02 });
    assert.ok(peak(out) > 0);
  });

  it('reports fractions in 0..1 and finishes on exactly 1', async () => {
    const seen = [];
    await phaseVocoder(toneInput(8000), 8, { windowSec: 0.02, onProgress: (f) => { seen.push(f); } });
    assert.ok(seen.length >= 2, `expected several progress calls, got ${seen.length}`);
    assert.equal(seen[0], 0);
    assert.equal(seen[seen.length - 1], 1);
    for (const f of seen) assert.ok(f >= 0 && f <= 1, `fraction ${f} out of range`);
  });

  it('reports monotonically', async () => {
    const seen = [];
    await phaseVocoder(toneInput(8000), 8, { windowSec: 0.02, onProgress: (f) => { seen.push(f); } });
    for (let i = 1; i < seen.length; i++) assert.ok(seen[i] >= seen[i - 1], 'progress went backwards');
  });

  it('awaits a promise-returning hook before continuing', async () => {
    const order = [];
    await phaseVocoder(toneInput(8000), 8, {
      windowSec: 0.02,
      onProgress: (f) => {
        order.push(`call:${f}`);
        return new Promise((resolve) => setImmediate(() => { order.push(`resume:${f}`); resolve(); }));
      },
    });
    // Every call must have resumed before the next one was made.
    for (let i = 0; i + 1 < order.length; i += 2) {
      assert.ok(order[i].startsWith('call:'));
      assert.ok(order[i + 1].startsWith('resume:'));
      assert.equal(order[i].slice(5), order[i + 1].slice(7));
    }
  });

  it('does not disturb the samples it reports on', async () => {
    const input = toneInput(8000);
    const quiet = await phaseVocoder(input, 8, { windowSec: 0.02 });
    const loud = await phaseVocoder(input, 8, { windowSec: 0.02, onProgress: async () => {} });
    assert.deepEqual(Array.from(loud.l), Array.from(quiet.l));
  });
});
