/**
 * Granular cloud — a port of `GranularKernel.swift` (swift-paul-stretch).
 *
 * Dense Hann-windowed grains are scattered from a scrub position that advances
 * through the source over the render. Unlike the stretch engines this keeps the
 * source's *timbre* intact while dissolving its rhythm, so it sits between
 * PaulStretch's wash and the original material.
 *
 * Every grain's identity — onset offset, source position, pitch and pan — comes
 * from a splitmix64-mixed per-grain seed, drawn in a fixed order. That is what
 * makes the engine reproducible and range-renderable: because a grain is a pure
 * function of its index, the Swift reference can render any slice of the output
 * on any core and get bit-identical audio, and this port gets the same answer.
 *
 * Verified against the Swift reference sample-for-sample (`tests/granular.test.js`).
 * Note the reference must be rendered with fades disabled: `StretchRenderer`
 * applies a 20 s fade envelope to one-shot renders, which is a renderer stage
 * rather than part of this engine.
 */

import { createBuffer, normalizeToPeak } from './buffer.js';
import { FastRNG, blockSeed } from './rng.js';
import { DEFAULT_SEED } from './stretch.js';

/**
 * Render a granular cloud of `targetSeconds` from `input`.
 *
 * Grains are laid on the output timeline at a fixed spacing
 * (`grainFrames / density`) and summed, then the whole cloud is normalised —
 * the per-grain gain is deliberately not compensated for density, because
 * overlapping grains from a correlated source sum somewhere between coherently
 * and incoherently and no single divisor is right for both.
 *
 * The source position sweeps linearly from the start of `input` to its end
 * across the render, so a long target from a short source lingers rather than
 * looping. `positionJitter` smears that sweep; `timeJitter` matters more than it
 * looks — at 0 grains fire on a perfectly rigid grid, and when the scrub lingers
 * near-identical content re-triggered at exactly `sampleRate / spacing` Hz reads
 * as a buzzy machine-gun. Jitter breaks the grid into an organic rain.
 *
 * @param {import('./buffer.js').StereoBuffer} input - Source audio. Mono
 *   buffers (where `r` is the same array as `l`) are handled without copying.
 * @param {number} targetSeconds - Length of the cloud to render, in seconds.
 *   Always produces at least one frame.
 * @param {object} [options]
 * @param {number} [options.grainSeconds=0.15] - Grain length in seconds. Floored
 *   at 5 ms, and at 64 frames after conversion, so a grain always has a usable
 *   Hann window.
 * @param {number} [options.density=8] - How many grains overlap at any instant.
 *   Values below 1 are treated as 1.
 * @param {number} [options.positionJitter=0.05] - Random source-position offset
 *   per grain, as a fraction of the whole source. Clamped to `0…1`.
 * @param {number} [options.timeJitter=0.75] - Random onset offset per grain, as
 *   a fraction of half the grain spacing. Clamped to `0…1`; 0 gives a rigid grid.
 * @param {number} [options.pitchSpread=0] - Random per-grain pitch, in ±
 *   semitones. Negative values are treated as 0.
 * @param {number} [options.basePitch=0] - Pitch applied to every grain, in
 *   semitones. Grains are rate-compensated so pitching does not shift where in
 *   the source they read from.
 * @param {number} [options.panSpread=0] - Random per-grain stereo position.
 *   Clamped to `0…1`; 0 keeps every grain centred, 1 spreads them hard L to R.
 * @param {bigint} [options.seed=DEFAULT_SEED] - Render seed. The same seed and
 *   parameters always produce identical output.
 * @param {(frac: number) => void|Promise<void>} [options.onProgress] - Called
 *   every 256 grains with completion in `0…1`, and once with 1 at the end. If it
 *   returns a promise it is awaited, which is how a browser caller yields to the
 *   event loop. Omit it and nothing is called or awaited.
 * @returns {Promise<import('./buffer.js').StereoBuffer>} The rendered cloud,
 *   peak-normalised to 0.92.
 *
 * @example
 * // Four seconds of soft cloud from a two-second source.
 * const cloud = await granular(source, 4);
 *
 * @example
 * // Dense, detuned, wide — and yielding to the browser as it goes.
 * const cloud = await granular(source, 30, {
 *   grainSeconds: 0.08,
 *   density: 16,
 *   positionJitter: 0.2,
 *   pitchSpread: 3,
 *   panSpread: 0.6,
 *   onProgress: (f) => new Promise((r) => setTimeout(() => r(showPercent(f)), 0)),
 * });
 */
export async function granular(input, targetSeconds, options = {}) {
  const sr = input.sampleRate;
  const inL = input.l;
  const inR = input.r;
  const inputLen = input.l.length;

  const grainSeconds = options.grainSeconds ?? 0.15;
  const density = options.density ?? 8;
  const positionJitter = options.positionJitter ?? 0.05;
  const timeJitter = options.timeJitter ?? 0.75;
  const pitchSpread = Math.max(0, options.pitchSpread ?? 0);
  const basePitch = options.basePitch ?? 0;
  const panSpread = Math.min(Math.max(options.panSpread ?? 0, 0), 1);
  const seed = options.seed ?? DEFAULT_SEED;
  const onProgress = options.onProgress;

  const outputLength = Math.max(1, Math.trunc(targetSeconds * sr));
  const out = createBuffer(outputLength, sr);
  if (inputLen <= 0) {
    if (onProgress) { const p = onProgress(1); if (p) await p; }
    return out;
  }

  const grainFrames = Math.max(64, Math.trunc(Math.max(0.005, grainSeconds) * sr));
  const spacing = grainFrames / Math.max(1, density);
  const jitterFrames = Math.min(Math.max(positionJitter, 0), 1) * inputLen;
  const onsetJitterFrames = Math.trunc(Math.min(Math.max(timeJitter, 0), 1) * spacing * 0.5);

  const env = new Float32Array(grainFrames);
  for (let i = 0; i < grainFrames; i++) {
    env[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (grainFrames - 1));
  }

  const outL = out.l;
  const outR = out.r;
  const totalGrains = Math.trunc(outputLength / spacing) + 1;

  let g = 0;
  for (;;) {
    const gridOnset = Math.trunc(g * spacing);
    if (gridOnset - onsetJitterFrames >= outputLength || gridOnset >= outputLength) break;

    // Fixed draw order: time, position, pitch, pan. The onset offset is
    // drawn unconditionally so a grain's identity never depends on which
    // range asked for it.
    const rng = new FastRNG(blockSeed(seed, g));
    // Swift's Int() truncates toward zero, so a negative offset must too —
    // Math.floor would be off by one for every leftward jitter.
    const onset = gridOnset + Math.trunc((rng.unit() * 2 - 1) * onsetJitterFrames);
    if (onset < outputLength && onset + grainFrames > 0) {
      const jitter = (rng.unit() * 2 - 1) * jitterFrames;
      const pitch = basePitch + (rng.unit() * 2 - 1) * pitchSpread;
      const pan = 0.5 + (rng.unit() * 2 - 1) * panSpread * 0.5;
      const rate = Math.pow(2, pitch / 12);
      const gainL = Math.cos((pan * Math.PI) / 2);
      const gainR = Math.sin((pan * Math.PI) / 2);

      // The scrub advances through the source across the cloud; the grain
      // is centred on it, rate-compensated so pitch doesn't shift where it
      // reads from.
      const scrub = outputLength > 1 ? onset / outputLength : 0;
      const srcCenter = scrub * Math.max(0, inputLen - 1) + jitter;
      const srcStart = srcCenter - grainFrames * rate * 0.5;

      const iStart = Math.max(0, -onset);
      const iEnd = Math.min(grainFrames, outputLength - onset);
      for (let i = iStart; i < iEnd; i++) {
        const sp = srcStart + i * rate;
        const s0 = Math.floor(sp);
        if (s0 >= 0 && s0 < inputLen) {
          const frac = sp - s0;
          const s1 = Math.min(s0 + 1, inputLen - 1);
          const e = env[i];
          const vL = (inL[s0] * (1 - frac) + inL[s1] * frac) * e;
          const vR = (inR[s0] * (1 - frac) + inR[s1] * frac) * e;
          const dest = onset + i;
          outL[dest] += vL * gainL;
          outR[dest] += vR * gainR;
        }
      }
    }

    g += 1;
    if (onProgress && (g & 255) === 0) {
      const p = onProgress(Math.min(1, g / totalGrains));
      if (p) await p;
    }
  }

  normalizeToPeak(out, 0.92);
  if (onProgress) { const p = onProgress(1); if (p) await p; }
  return out;
}
