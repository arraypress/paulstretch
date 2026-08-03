/**
 * Spectral freeze — a port of `FreezeKernel.swift` (swift-paul-stretch).
 *
 * The magnitude spectrum is captured at a single instant in the source and
 * resynthesised forever with fresh random phase every hop: one frozen
 * moment, shimmering indefinitely. Nothing of the source's movement
 * survives, which is exactly the point — it is the most "suspended" of the
 * engines.
 *
 * Two knobs shape it:
 *  - `smear` box-blurs the captured magnitudes, washing tonal peaks toward
 *    noise (radius ≈ smear × 50 bins — deliberately small, since a radius
 *    proportional to the window flattens everything into white noise).
 *  - `scan` drifts the capture point through the source over the render, so
 *    the frozen spectrum slowly morphs instead of standing perfectly still.
 *
 * Phases come from a per-hop seed (`blockSeed`), so output is reproducible
 * and any hop can be rendered independently — the same property that lets
 * the Swift version render chunked and multicore.
 */

import { createBuffer, normalizeToPeak } from './buffer.js';
import { fft } from './fft.js';
import { FastRNG, blockSeed, nextPow2 } from './rng.js';
import { DEFAULT_SEED } from './stretch.js';

/**
 * Moving-average blur of the lower-half magnitude spectrum, in place.
 *
 * Every bin averages a window of the SNAPSHOT taken before the loop starts,
 * never of the partially-blurred array it is writing into. Reading the live
 * array would feed each result into the next bin's window — a cascading,
 * direction-dependent smear that diverges further from Swift's `boxBlur` the
 * wider the radius gets.
 *
 * Bin 0 (DC) is left alone, and the window is clamped to `1…half - 1`, so the
 * blur never drags energy across DC or the Nyquist bin.
 *
 * @param {Float32Array} a - Magnitudes. Bins `1…half - 1` are overwritten.
 * @param {number} half - Half the window size, i.e. the Nyquist bin index.
 * @param {number} radius - Blur radius, in bins.
 * @returns {void}
 */
function boxBlur(a, half, radius) {
  const src = a.slice();
  for (let k = 1; k < half; k++) {
    const lo = Math.max(1, k - radius);
    const hi = Math.min(half - 1, k + radius);
    let s = 0;
    for (let j = lo; j <= hi; j++) s += src[j];
    a[k] = s / (hi - lo + 1);
  }
}

/**
 * Windows the source at a normalised position and writes its (optionally
 * smeared) magnitude spectrum into `magL` / `magR`.
 *
 * Shared by the cached static capture and the per-hop scanning capture, so
 * both paths smear identically. The window is placed by truncating — not
 * rounding — `positionNorm × (inputLen - windowSize)`, which is what Swift's
 * `Int(_:)` conversion does; changing it would shift the capture by a frame
 * and invalidate every seeded render.
 *
 * @param {number} positionNorm - Capture point, `0…1` through the source.
 * @param {Float32Array} inL - Source left channel.
 * @param {Float32Array} inR - Source right channel (may be the same array).
 * @param {number} inputLen - Source length in frames.
 * @param {number} windowSize - STFT window size in frames (a power of two).
 * @param {number} half - Half the window size.
 * @param {Float32Array} hann - The analysis window.
 * @param {number} smear - Magnitude blur amount, `0…1`.
 * @param {Float32Array} magL - Destination for the left magnitudes.
 * @param {Float32Array} magR - Destination for the right magnitudes.
 * @returns {void}
 */
function capture(positionNorm, inL, inR, inputLen, windowSize, half, hann, smear, magL, magR) {
  const center = Math.trunc(positionNorm * Math.max(0, inputLen - windowSize));
  const rL = new Float32Array(windowSize), iL = new Float32Array(windowSize);
  const rR = new Float32Array(windowSize), iR = new Float32Array(windowSize);
  for (let i = 0; i < windowSize; i++) {
    const idx = center + i;
    const w = hann[i];
    if (idx >= 0 && idx < inputLen) { rL[i] = inL[idx] * w; rR[i] = inR[idx] * w; }
  }
  fft(rL, iL, false);
  fft(rR, iR, false);
  for (let k = 0; k < windowSize; k++) {
    magL[k] = Math.sqrt(rL[k] * rL[k] + iL[k] * iL[k]);
    magR[k] = Math.sqrt(rR[k] * rR[k] + iR[k] * iR[k]);
  }
  if (smear > 0.01) {
    // Gentle: a few bins at low smear (keeps tonal peaks), up to ~50 bins at
    // full smear (washy). A radius proportional to the window would flatten
    // the whole spectrum into white noise.
    const radius = Math.max(1, Math.trunc(smear * 50));
    boxBlur(magL, half, radius);
    boxBlur(magR, half, radius);
  }
}

/**
 * Freeze one instant of `input` and resynthesise it for `targetSeconds`.
 *
 * The result is a pad, not a stretch: the source's magnitude spectrum is
 * sampled at a single point and re-emitted every hop with brand-new random
 * phases, which is what turns a static spectrum into something that breathes
 * instead of ringing like a held chord.
 *
 * Each hop seeds its own generator from `blockSeed(seed, hopIndex)`, so the
 * output depends only on the seed and the parameters — never on how the render
 * was scheduled. That is why the Swift reference can render this across cores
 * and in chunks and still get bit-identical audio, and it is why the same
 * arguments here always produce the same samples.
 *
 * The finished pad is peak-normalised to 0.92 using the louder channel, so the
 * stereo image survives normalisation.
 *
 * @param {import('./buffer.js').StereoBuffer} input - The source to freeze.
 *   Sources shorter than 32 frames are too short to window and yield silence.
 * @param {number} targetSeconds - Duration to synthesise. The output is never
 *   shorter than one window.
 * @param {Object} [options]
 * @param {number} [options.positionNorm=0.5] - Capture point, `0…1` through
 *   the source. Clamped.
 * @param {number} [options.smear=0] - Magnitude box-blur, `0…1`. Low keeps
 *   tonal peaks; high washes toward coloured noise. Values at or below 0.01
 *   skip the blur entirely. Clamped.
 * @param {number} [options.scan=0] - How far the capture point drifts through
 *   the source over the render, `0…1`. `0` is the classic static freeze;
 *   above zero each hop re-captures at `positionNorm + scan × elapsed`, so the
 *   spectrum slowly morphs. Clamped.
 * @param {number} [options.windowSec=0.25] - STFT window length in seconds,
 *   rounded up to a power-of-two frame count. Longer windows freeze more
 *   smoothly and blur transients further.
 * @param {bigint} [options.seed=DEFAULT_SEED] - Render seed.
 * @param {(frac: number) => (void | Promise<void>)} [options.onProgress] -
 *   Called every 32 hops with the completed fraction, and once with `1` at the
 *   end. If it returns a promise it is awaited, which is how a browser caller
 *   yields to the event loop. Omit it and nothing is called or awaited.
 * @returns {Promise<import('./buffer.js').StereoBuffer>} The frozen pad,
 *   peak-normalised to 0.92.
 *
 * @example
 * // Four minutes of pad from the middle of a short sample.
 * const pad = await freeze(source, 240, { positionNorm: 0.5, smear: 0.3 });
 *
 * @example
 * // A freeze that slowly scans forward through the source, reporting
 * // progress and yielding so a UI stays responsive.
 * const pad = await freeze(source, 120, {
 *   positionNorm: 0.1,
 *   scan: 0.8,
 *   windowSec: 0.5,
 *   onProgress: (frac) => new Promise((r) => setTimeout(() => {
 *     bar.value = frac;
 *     r();
 *   }, 0)),
 * });
 */
export async function freeze(input, targetSeconds, options = {}) {
  const sr = input.sampleRate;
  const inL = input.l;
  const inR = input.r;
  const inputLen = input.l.length;

  const positionNorm = Math.min(Math.max(options.positionNorm ?? 0.5, 0), 1);
  const smear = Math.min(Math.max(options.smear ?? 0, 0), 1);
  const scan = Math.min(Math.max(options.scan ?? 0, 0), 1);
  const windowSec = options.windowSec ?? 0.25;
  const seed = options.seed ?? DEFAULT_SEED;
  const onProgress = options.onProgress;

  const windowSize = nextPow2(Math.trunc(windowSec * sr));
  const half = windowSize >> 1;
  const hop = windowSize >> 2;
  const outputLength = Math.max(windowSize, Math.trunc(targetSeconds * sr));

  const out = createBuffer(outputLength, sr);
  // Too short to window — Swift returns nil here; silence is the analogue.
  if (inputLen < 32) {
    if (onProgress) { const p = onProgress(1); if (p) await p; }
    return out;
  }

  // SYMMETRIC Hann (denominator windowSize - 1): only the PERIODIC form
  // (/windowSize) sums exactly flat when squared at hop N/4; this one leaves
  // O(1/N) ripple (~-80 dB at N = 8192), inaudible and hidden by the peak
  // normalise. Kept symmetric because changing it would invalidate every
  // seeded render ever made with this library.
  const hann = new Float32Array(windowSize);
  for (let i = 0; i < windowSize; i++) {
    hann[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (windowSize - 1));
  }

  const magL = new Float32Array(windowSize);
  const magR = new Float32Array(windowSize);
  // Static freeze captures once up front; scanning re-captures per hop below.
  if (scan <= 0) {
    capture(positionNorm, inL, inR, inputLen, windowSize, half, hann, smear, magL, magR);
  }

  const lastBlock = Math.trunc((outputLength - half - 1) / hop);
  const outL = out.l;
  const outR = out.r;

  const rL = new Float32Array(windowSize), iL = new Float32Array(windowSize);
  const rR = new Float32Array(windowSize), iR = new Float32Array(windowSize);
  const sML = new Float32Array(scan > 0 ? windowSize : 1);
  const sMR = new Float32Array(scan > 0 ? windowSize : 1);

  for (let b = 0; b <= lastBlock; b++) {
    const outputPos = b * hop;

    let mL = magL, mR = magR;
    if (scan > 0) {
      // Capture position is a pure function of the hop index, so any
      // partition of the render agrees on it.
      const tau = outputPos / outputLength;
      const pos = Math.min(Math.max(positionNorm + scan * tau, 0), 1);
      capture(pos, inL, inR, inputLen, windowSize, half, hann, smear, sML, sMR);
      mL = sML; mR = sMR;
    }

    // Fresh random phase per hop. L and R alternate per bin — the exact
    // RNG consumption order the Swift kernel uses.
    const rng = new FastRNG(blockSeed(seed, b));
    rL[0] = mL[0]; iL[0] = 0; rR[0] = mR[0]; iR[0] = 0;
    for (let k = 1; k < half; k++) {
      const aL = rng.unit() * 2 * Math.PI;
      const aR = rng.unit() * 2 * Math.PI;
      const vL = mL[k], vR = mR[k];
      rL[k] = vL * Math.cos(aL); iL[k] = vL * Math.sin(aL);
      rR[k] = vR * Math.cos(aR); iR[k] = vR * Math.sin(aR);
      rL[windowSize - k] = rL[k]; iL[windowSize - k] = -iL[k];
      rR[windowSize - k] = rR[k]; iR[windowSize - k] = -iR[k];
    }
    rL[half] = 0; iL[half] = 0; rR[half] = 0; iR[half] = 0;

    fft(rL, iL, true);
    fft(rR, iR, true);

    const iEnd = Math.min(windowSize, outputLength - outputPos);
    for (let i = 0; i < iEnd; i++) {
      const w = hann[i];
      outL[outputPos + i] += rL[i] * w;
      outR[outputPos + i] += rR[i] * w;
    }

    // Progress is a caller-supplied hook rather than a wall-clock yield: a
    // Node caller that passes nothing pays nothing and never leaves the loop.
    // Every 32 hops is the browser original's check interval.
    if (onProgress && (b & 31) === 0) {
      const p = onProgress(b / (lastBlock + 1));
      if (p) await p;
    }
  }

  // Swift's SpectralFreezer peak-normalises the finished pad to 0.92, using
  // the louder of the two channels so the stereo image is preserved.
  normalizeToPeak(out, 0.92);

  if (onProgress) { const p = onProgress(1); if (p) await p; }
  return out;
}
