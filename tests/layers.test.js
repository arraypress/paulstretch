/**
 * Layer-recipe parity: `LAYER_PRESETS` / `pickLayers` vs `LayerPreset.swift`
 * (swift-paul-stretch), plus the two derivations from `RenderPlan.swift`.
 *
 * These recipes are plain data, which is exactly why they need a test — a
 * wrong gain or a dropped pitch offset changes every layered render and
 * nothing else would catch it. The expected values below are transcribed from
 * `Sources/PaulStretch/Enums/LayerPreset.swift`; if that file changes, this
 * test is the thing that should fail.
 *
 * ORDER MATTERS. Each layer's phase seed is derived from its INDEX
 * (`seed + i * 0x9E3779B97F4A7C15`), so reordering a recipe silently changes
 * the audio even though the same layers are present.
 */
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { LAYER_PRESETS, pickLayers, layerSeed, layerPitch } from '../src/layers.js';

/** [scale, gain, pitchSemitones] — pitch defaults to 0 when omitted. */
const SWIFT = {
  off: null,
  subtle: [
    [0.7, 0.45, 0],
    [1.0, 0.75, 0],
    [1.4, 0.45, 0],
  ],
  standard: [
    [0.5, 0.55, 0],
    [1.0, 0.70, 0],
    [2.0, 0.50, 0],
  ],
  lush: [
    [0.25, 0.40, 0],
    [0.5, 0.55, 0],
    [1.0, 0.70, 0],
    [2.0, 0.55, 0],
    [4.0, 0.40, 0],
  ],
  shimmer: [
    [0.5, 0.55, 0],
    [1.0, 0.70, 0],
    [1.0, 0.45, 12],
  ],
  shimmerDeep: [
    [0.5, 0.50, -12],
    [1.0, 0.70, 0],
    [1.0, 0.45, 12],
  ],
};

/** The golden-ratio stride the Swift plan builder uses. */
const GOLDEN = 0x9e3779b97f4a7c15n;
/** `PaulStretcher.defaultSeed` — the shared default render seed. */
const DEFAULT_SEED = 0x2545f4914f6cdd1dn;

/** Layers as `[scale, gain, pitch]` triples, for comparison with SWIFT. */
function triples(layers) {
  return layers.map((l) => [l.scale, l.gain, l.pitchSemitones ?? 0]);
}

describe('layer recipes match the Swift reference', () => {
  for (const [mode, want] of Object.entries(SWIFT)) {
    it(`${mode} matches scale/gain/pitch in order`, () => {
      const got = pickLayers(mode);

      if (want === null) {
        assert.equal(got, null, 'off returns null (single unlayered pass)');
        return;
      }

      assert.notEqual(got, null, `${mode} returned null, expected a recipe`);
      assert.equal(got.length, want.length, `${mode} layer count`);
      assert.deepEqual(triples(got), want);
    });
  }

  it('LAYER_PRESETS carries the same data as pickLayers', () => {
    for (const [mode, want] of Object.entries(SWIFT)) {
      if (want === null) {
        assert.equal(LAYER_PRESETS[mode], null);
      } else {
        assert.deepEqual(triples(LAYER_PRESETS[mode]), want);
      }
    }
  });

  it('exposes exactly the six documented modes', () => {
    assert.deepEqual(Object.keys(LAYER_PRESETS), [
      'off', 'subtle', 'standard', 'lush', 'shimmer', 'shimmerDeep',
    ]);
  });
});

describe('the shimmer recipes actually carry pitch offsets', () => {
  // Guards the whole point of these two modes: if the pitch fields were ever
  // dropped they would silently degrade into duplicates of `standard`.
  it('shimmer has exactly one octave-up voice', () => {
    const shimmer = pickLayers('shimmer') ?? [];
    assert.equal(shimmer.filter((l) => l.pitchSemitones === 12).length, 1);
  });

  it('shimmerDeep has an octave up AND an octave down', () => {
    const deep = pickLayers('shimmerDeep') ?? [];
    assert.ok(deep.some((l) => l.pitchSemitones === 12));
    assert.ok(deep.some((l) => l.pitchSemitones === -12));
  });

  it('the same-pitch recipes carry no offsets', () => {
    for (const mode of ['subtle', 'standard', 'lush']) {
      const layers = pickLayers(mode) ?? [];
      assert.ok(layers.every((l) => (l.pitchSemitones ?? 0) === 0), mode);
    }
  });

  it('shimmerDeep is not a re-spelling of shimmer', () => {
    assert.notDeepEqual(triples(pickLayers('shimmerDeep')), triples(pickLayers('shimmer')));
  });
});

describe('recipe ordering', () => {
  // Layer index feeds the phase seed, so these orders are load-bearing.
  it('every recipe is ordered slow-drifting last (scales ascending)', () => {
    for (const mode of ['subtle', 'standard', 'lush']) {
      const scales = pickLayers(mode).map((l) => l.scale);
      const sorted = [...scales].sort((a, b) => a - b);
      assert.deepEqual(scales, sorted, mode);
    }
  });

  it('the shimmer voice is the last layer in both shimmer recipes', () => {
    for (const mode of ['shimmer', 'shimmerDeep']) {
      const layers = pickLayers(mode);
      assert.equal(layers[layers.length - 1].pitchSemitones, 12, mode);
    }
  });
});

describe('pickLayers', () => {
  it('returns null only for off', () => {
    assert.equal(pickLayers('off'), null);
    for (const mode of ['subtle', 'standard', 'lush', 'shimmer', 'shimmerDeep']) {
      assert.notEqual(pickLayers(mode), null, mode);
    }
  });

  it('falls back to standard for an unknown mode', () => {
    const standard = triples(pickLayers('standard'));
    for (const mode of ['nonsense', '', 'OFF', 'Standard', undefined, null, 42]) {
      assert.deepEqual(triples(pickLayers(mode)), standard, String(mode));
    }
  });

  it('is not fooled by inherited Object properties', () => {
    // `LAYER_PRESETS['constructor']` would otherwise resolve to a function.
    const standard = triples(pickLayers('standard'));
    for (const mode of ['constructor', 'toString', '__proto__', 'hasOwnProperty']) {
      assert.deepEqual(triples(pickLayers(mode)), standard, mode);
    }
  });

  it('returns a fresh copy each call', () => {
    const a = pickLayers('lush');
    const b = pickLayers('lush');
    assert.notEqual(a, b);
    assert.notEqual(a[0], b[0]);
    assert.deepEqual(a, b);
  });

  it('mutating the result cannot corrupt the shared table', () => {
    const mine = pickLayers('shimmer');
    mine[2].scale = 2.0;
    mine.push({ scale: 8, gain: 0.1 });
    assert.deepEqual(triples(pickLayers('shimmer')), SWIFT.shimmer);
    assert.equal(LAYER_PRESETS.shimmer[2].scale, 1.0);
  });

  it('every layer carries a finite scale and gain', () => {
    for (const mode of Object.keys(SWIFT)) {
      for (const layer of pickLayers(mode) ?? []) {
        assert.ok(Number.isFinite(layer.scale) && layer.scale > 0, mode);
        assert.ok(Number.isFinite(layer.gain) && layer.gain > 0, mode);
      }
    }
  });
});

describe('LAYER_PRESETS is immutable', () => {
  it('rejects adding a mode', () => {
    assert.throws(() => { LAYER_PRESETS.extraLush = []; }, TypeError);
  });

  it('rejects replacing a recipe', () => {
    assert.throws(() => { LAYER_PRESETS.standard = []; }, TypeError);
  });

  it('rejects editing a layer in place', () => {
    assert.throws(() => { LAYER_PRESETS.subtle[0].gain = 1; }, TypeError);
  });

  it('rejects appending a layer', () => {
    assert.throws(() => { LAYER_PRESETS.subtle.push({ scale: 1, gain: 1 }); }, TypeError);
  });
});

describe('layerSeed', () => {
  it('leaves layer 0 on the base seed', () => {
    // A one-layer recipe must render identically to an unlayered pass.
    assert.equal(layerSeed(DEFAULT_SEED, 0), DEFAULT_SEED);
    assert.equal(layerSeed(0n, 0), 0n);
  });

  it('matches the Swift stride for the default seed', () => {
    assert.equal(layerSeed(DEFAULT_SEED, 0), 2685821657736338717n);
    assert.equal(layerSeed(DEFAULT_SEED, 1), 14086536477059537202n);
    assert.equal(layerSeed(DEFAULT_SEED, 2), 7040507222673184071n);
    assert.equal(layerSeed(DEFAULT_SEED, 3), 18441222041996382556n);
    assert.equal(layerSeed(DEFAULT_SEED, 4), 11395192787610029425n);
  });

  it('is deterministic — same seed twice, identical result', () => {
    for (let i = 0; i < 8; i++) {
      assert.equal(layerSeed(DEFAULT_SEED, i), layerSeed(DEFAULT_SEED, i));
    }
  });

  it('gives every layer of a recipe a distinct seed', () => {
    // The whole point: colliding seeds render bit-identical layers that sum
    // coherently, so the stack gets louder instead of thicker.
    for (const mode of ['subtle', 'standard', 'lush', 'shimmer', 'shimmerDeep']) {
      const layers = pickLayers(mode);
      const seeds = new Set(layers.map((_, i) => layerSeed(DEFAULT_SEED, i)));
      assert.equal(seeds.size, layers.length, mode);
    }
  });

  it('different base seeds give different layer seeds', () => {
    for (let i = 0; i < 5; i++) {
      assert.notEqual(layerSeed(DEFAULT_SEED, i), layerSeed(DEFAULT_SEED + 1n, i));
    }
  });

  it('wraps to 64 bits like Swift &+ / &*', () => {
    assert.equal(layerSeed(0xffffffffffffffffn, 1), GOLDEN - 1n);
    assert.equal(layerSeed(0n, 3), BigInt.asUintN(64, 3n * GOLDEN));
  });

  it('always lands in [0, 2^64)', () => {
    const bases = [0n, 1n, DEFAULT_SEED, 0xffffffffffffffffn, 0x8000000000000000n];
    for (const base of bases) {
      for (let i = 0; i < 6; i++) {
        const s = layerSeed(base, i);
        assert.ok(s >= 0n && s < 1n << 64n, `${base}/${i}`);
      }
    }
  });

  it('accepts a large layer index', () => {
    assert.equal(layerSeed(DEFAULT_SEED, 1000), BigInt.asUintN(64, DEFAULT_SEED + 1000n * GOLDEN));
  });

  it('requires a bigint base seed', () => {
    assert.throws(() => layerSeed(12345, 1), TypeError);
  });
});

describe('layerPitch', () => {
  it('stacks the layer offset on the global shift', () => {
    assert.equal(layerPitch(0, { scale: 1, gain: 0.45, pitchSemitones: 12 }), 12);
    assert.equal(layerPitch(-5, { scale: 1, gain: 0.45, pitchSemitones: 12 }), 7);
    assert.equal(layerPitch(7, { scale: 0.5, gain: 0.5, pitchSemitones: -12 }), -5);
  });

  it('treats a missing offset as 0', () => {
    assert.equal(layerPitch(-5, { scale: 1, gain: 0.7 }), -5);
    assert.equal(layerPitch(0, { scale: 1, gain: 0.7 }), 0);
    assert.equal(layerPitch(3.5, { scale: 1, gain: 0.7, pitchSemitones: undefined }), 3.5);
  });

  it('keeps a transposed shimmer an octave above its wash', () => {
    // The reason offsets stack rather than replace: transposing the drone must
    // move every voice together or the interval collapses.
    const layers = pickLayers('shimmer');
    for (const base of [-12, -5, 0, 3, 7.5]) {
      const wash = layerPitch(base, layers[1]);
      const voice = layerPitch(base, layers[2]);
      assert.equal(voice - wash, 12);
    }
  });

  it('spans two octaves across shimmerDeep', () => {
    const layers = pickLayers('shimmerDeep');
    const pitches = layers.map((l) => layerPitch(0, l));
    assert.equal(Math.max(...pitches) - Math.min(...pitches), 24);
  });

  it('handles fractional global shifts without rounding', () => {
    assert.equal(layerPitch(0.5, { scale: 1, gain: 0.45, pitchSemitones: 12 }), 12.5);
  });
});
