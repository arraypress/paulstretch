/**
 * PaulStretch — a port of `StretchKernel.swift` (swift-paul-stretch).
 *
 * PaulStretch (Paul Nasca, 2008) is a time-stretching algorithm built around
 * STFT phase randomisation rather than the more common phase-vocoder approach.
 * For ambient material — pads, drones, long evolving timbres — it produces a
 * far more natural "stretched-to-infinity" wash than phase-vocoder methods.
 *
 * Pipeline:
 *  1. Optionally resample the input, which is how pitch is shifted (see
 *     `resampleChannel`) — the stretch ratio is rescaled to compensate so the
 *     output length is unchanged.
 *  2. Window the input (Hann), step forward in INPUT space by `stride / ratio`.
 *  3. FFT each window into magnitude + phase.
 *  4. Replace phases with random angles (mirrored for the conjugate half so
 *     the inverse FFT yields a real signal).
 *  5. Inverse FFT, re-window, overlap-add into OUTPUT space at full stride.
 *
 * The result is a buffer whose duration is `ratio ×` longer than the input,
 * with the same long-term spectral envelope but no time-locked transients.
 *
 * Every window's phases come from a splitmix64-mixed per-window seed, so the
 * same source + options + seed produce bit-identical output — and output
 * identical to the Swift library's. That per-window independence is also what
 * lets a render be split across workers or time chunks without seams.
 */

import { fft } from './fft.js';
import { FastRNG, blockSeed, nextPow2 } from './rng.js';
import { createBuffer, normalizeToPeak } from './buffer.js';

/** STFT window length used when the caller doesn't pick one, in seconds. */
const DEFAULT_WINDOW_SEC = 0.25;

/**
 * The seed every render uses unless the caller supplies one.
 *
 * Matches `PaulStretcher.defaultSeed` in the Swift library — renders made with
 * the default seed are identical across both implementations, which is what
 * the parity fingerprints in `tests/stretch.test.js` pin.
 *
 * @type {bigint}
 *
 * @example
 * await stretch(source, 8);                        // uses DEFAULT_SEED
 * await stretch(source, 8, { seed: DEFAULT_SEED }); // same render, explicitly
 */
export const DEFAULT_SEED = 0x2545f4914f6cdd1dn;

/** Golden-ratio constant behind the spectral-drift read position. */
const PHI = 1.618033988749895;

/** The frame size (in samples) the onset curve is computed over. */
const ONSET_FRAME_SIZE = 512;

/**
 * Windows between `onProgress` calls.
 *
 * Matches the Swift kernel's `onBlocksDone` cadence (every 64 windows). The
 * browser original used a 40 ms wall-clock timer instead; a fixed window count
 * is both deterministic and free of a `performance.now()` dependency.
 */
const PROGRESS_BLOCKS = 64;

/** Clamp to the 0..1 range every normalised parameter here lives in. */
function clamp01(v) {
  return Math.max(0, Math.min(1, v));
}

/**
 * Wraps a phase difference to (-pi, pi] so blending takes the short way round
 * rather than sweeping the long arc.
 *
 * @param {number} x - A phase difference in radians.
 * @returns {number} The equivalent difference in (-pi, pi].
 */
function wrapPi(x) {
  let a = x % (2 * Math.PI);
  if (a > Math.PI) a -= 2 * Math.PI;
  if (a <= -Math.PI) a += 2 * Math.PI;
  return a;
}

/**
 * A moving average wide enough to cover the new sample spacing, applied only
 * when decimating. Taking every 1/factor-th sample of full-band audio folds
 * everything above the new Nyquist back down as aliasing.
 *
 * The accumulator is kept at float32 (`Math.fround`) to match Swift's `Float`
 * running sum sample-for-sample — a float64 accumulator would drift away from
 * the reference over a long channel, which is exactly what the parity
 * fingerprints would catch.
 *
 * @param {Float32Array} x - Source channel.
 * @param {number} n - Number of frames in `x`.
 * @param {number} factor - Length multiplier. Values >= 1 are interpolation
 *   and need no pre-filter, so `x` is returned untouched.
 * @returns {Float32Array} The filtered channel, or `x` itself when no filter
 *   is required.
 */
function prefilter(x, n, factor) {
  if (factor >= 1) return x;
  const width = Math.max(2, Math.round(1 / factor));
  if (width <= 1) return x;
  const out = new Float32Array(n);
  let running = 0;
  for (let i = 0; i < n; i++) {
    running = Math.fround(running + x[i]);
    if (i >= width) running = Math.fround(running - x[i - width]);
    out[i] = running / Math.min(i + 1, width);
  }
  return out;
}

/**
 * Resamples a channel, changing its length by `factor` and its pitch by
 * `1 / factor`.
 *
 * Catmull-Rom rather than linear: linear interpolation of a waveform is a
 * lowpass whose corner moves with the fractional offset, which is its own
 * source of shimmer — precisely the class of artefact this is here to remove.
 *
 * Decimation (`factor < 1`, i.e. shifting UP) is pre-filtered — see
 * `prefilter`.
 *
 * @param {Float32Array} x - Source channel.
 * @param {number} n - Number of frames in `x`.
 * @param {number} outCount - Number of frames to produce.
 * @param {number} factor - Length multiplier (`outCount / n`, give or take
 *   rounding).
 * @returns {Float32Array} The resampled channel.
 */
function resampleChannel(x, n, outCount, factor) {
  const src = prefilter(x, n, factor);
  const out = new Float32Array(outCount);
  const step = (n - 1) / (outCount - 1);
  for (let i = 0; i < outCount; i++) {
    const pos = i * step;
    const k = Math.trunc(pos);
    const t = pos - k;
    const p0 = src[Math.max(k - 1, 0)];
    const p1 = src[Math.min(k, n - 1)];
    const p2 = src[Math.min(k + 1, n - 1)];
    const p3 = src[Math.min(k + 2, n - 1)];
    out[i] = 0.5 * ((2 * p1)
                  + (-p0 + p2) * t
                  + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                  + (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t);
  }
  return out;
}

/**
 * Compute a per-frame onset-strength curve via half-wave-rectified energy
 * gradient. Cheaper than a full spectral-flux pass (no FFT) and works well
 * for the use case here — we don't need pinpoint transient detection, just
 * "is this region energy-rising-fast" so the stretcher can ease off the
 * phase scramble.
 *
 * Result is normalised to 0..1 across the whole input, smoothed with a
 * 3-frame kernel so a one-frame spike doesn't cause a click-pattern in the
 * effective randomness.
 *
 * @param {Float32Array} inL - Left channel of the analysed source.
 * @param {Float32Array} inR - Right channel (may be the same array as `inL`).
 * @param {number} frameSize - Frames per onset frame.
 * @returns {Float32Array} Onset strength per frame, normalised 0..1. All zero
 *   for silent or constant-energy input.
 */
function computeOnsetCurve(inL, inR, frameSize) {
  const numFrames = Math.max(1, Math.floor(inL.length / frameSize));
  const energies = new Float32Array(numFrames);
  for (let f = 0; f < numFrames; f++) {
    let e = 0;
    const start = f * frameSize;
    const end = Math.min(start + frameSize, inL.length);
    for (let i = start; i < end; i++) {
      const s = (inL[i] + inR[i]) * 0.5;
      e += s * s;
    }
    energies[f] = Math.sqrt(e / Math.max(1, end - start));
  }
  // Half-wave rectified gradient = "energy is rising" only.
  const raw = new Float32Array(numFrames);
  for (let f = 1; f < numFrames; f++) {
    raw[f] = Math.max(0, energies[f] - energies[f - 1]);
  }
  // Three-frame box smooth — keeps single-frame spikes from causing
  // randomness flicker but doesn't blur out the onset's leading edge.
  const smoothed = new Float32Array(numFrames);
  for (let f = 0; f < numFrames; f++) {
    let acc = raw[f];
    let count = 1;
    if (f > 0) { acc += raw[f - 1]; count++; }
    if (f < numFrames - 1) { acc += raw[f + 1]; count++; }
    smoothed[f] = acc / count;
  }
  // Normalise to [0, 1]. If the input is dead silent / constant-energy,
  // return a zero curve so the stretcher operates as if no onsets exist.
  let maxOnset = 0;
  for (let f = 0; f < numFrames; f++) if (smoothed[f] > maxOnset) maxOnset = smoothed[f];
  if (maxOnset > 0) {
    for (let f = 0; f < numFrames; f++) smoothed[f] /= maxOnset;
  }
  return smoothed;
}

/**
 * Phase randomisation with adjustable amount.
 *   amount = 1 → fully random phase (classic PaulStretch wash)
 *   amount = 0 → original phase preserved (tonal content survives)
 *   amount = 0.5 → each bin rotated by a random angle in ±π·amount
 *
 * Magnitude is always preserved; only the phase changes. The upper half of
 * the spectrum is mirrored as the complex conjugate so the inverse FFT yields
 * a real-valued signal.
 *
 * `continuity` blends in the previous window's angle for the same bin, taking
 * the short way round the circle. `prevRng` must be the same object across
 * the L and R calls so its stream stays in lockstep with `rng` — including in
 * the partial-randomness branch, where the draw is discarded but must still
 * happen. The draw ORDER differs between the two branches (`rng` first at full
 * randomness, `prevRng` first below it); that is what the Swift kernel does and
 * changing it changes every sample of every seeded render.
 *
 * @param {Float32Array} real - Real spectrum, mutated in place.
 * @param {Float32Array} imag - Imaginary spectrum, mutated in place.
 * @param {number} n - Transform size (`real.length`).
 * @param {number} amount - Randomisation amount, 0..1.
 * @param {FastRNG} rng - This window's generator.
 * @param {FastRNG|null} prevRng - The previous window's generator, replayed in
 *   lockstep, or `null` when there is no predecessor to blend with.
 * @param {number} continuity - How much of the previous window's angle to
 *   carry, 0..1.
 * @returns {void}
 */
function randomizePhases(real, imag, n, amount, rng, prevRng, continuity) {
  const half = n >> 1;
  if (amount >= 1) {
    for (let k = 1; k < half; k++) {
      const re = real[k];
      const im = imag[k];
      const mag = Math.sqrt(re * re + im * im);
      let angle = rng.unit() * 2 * Math.PI;
      // Carry a fraction of the previous window's phase for this bin.
      if (continuity > 0 && prevRng) {
        const prev = prevRng.unit() * 2 * Math.PI;
        angle = prev + (1 - continuity) * wrapPi(angle - prev);
      }
      const rv = mag * Math.cos(angle);
      const iv = mag * Math.sin(angle);
      real[k] = rv;
      imag[k] = iv;
      real[n - k] = rv;
      imag[n - k] = -iv;
    }
  } else if (amount > 0) {
    // Partial: rotate each bin by a random delta in ±π·amount, preserving
    // magnitude. At amount=0 this is a no-op (delta range collapses to 0).
    const maxDelta = Math.PI * amount;
    for (let k = 1; k < half; k++) {
      const re = real[k];
      const im = imag[k];
      const mag = Math.sqrt(re * re + im * im);
      const oldAngle = Math.atan2(im, re);
      // Keep the streams aligned even though partial randomisation is
      // already anchored to the source phase and needs no continuity.
      if (prevRng) prevRng.unit();
      const delta = (rng.unit() * 2 - 1) * maxDelta;
      const newAngle = oldAngle + delta;
      const rv = mag * Math.cos(newAngle);
      const iv = mag * Math.sin(newAngle);
      real[k] = rv;
      imag[k] = iv;
      real[n - k] = rv;
      imag[n - k] = -iv;
    }
  }
  // amount = 0: leave the spectrum untouched.
  imag[0] = 0;
  imag[half] = 0;
}

/**
 * Stretch `input` by `ratio`, returning a buffer roughly `ratio ×` longer and
 * peak-normalised to 0.92.
 *
 * Ratios of 1.001 or below return a copy of the input unchanged — the kernel
 * requires a real stretch, and near-unity ratios are not worth the artefacts.
 *
 * The function is `async` only so that a caller with a UI to keep responsive
 * can yield from `onProgress`. With no `onProgress` nothing is called and
 * nothing is awaited, so a Node or worker caller pays no scheduling cost at
 * all and the whole render runs in one turn.
 *
 * @param {import('./buffer.js').StereoBuffer} input - The source audio. Mono
 *   sources (where `r` is the same array as `l`) are handled without copying.
 * @param {number} ratio - The stretch factor (`8` → eight times longer).
 * @param {object} [options]
 * @param {number} [options.windowSec=0.25] - PaulStretch window size in
 *   seconds, rounded up to a power-of-two frame count. Bigger = washier, more
 *   diffuse, and slower to respond to the source's own changes.
 * @param {number} [options.phaseRandomness=1] - Phase randomisation amount in
 *   0..1. `1` is full randomness (the classic PaulStretch noise wash); `0`
 *   preserves the original phase, so tonal content stays recognisable and
 *   there is no wash at all.
 * @param {number} [options.pitchSemitones=0] - Semitone pitch shift, applied
 *   by RESAMPLING THE INPUT (not by remapping FFT bins — see
 *   `resampleChannel`). Decoupled from output length: the stretch ratio is
 *   rescaled to compensate, so the buffer's duration is preserved.
 * @param {number} [options.onsetSensitivity=0] - Onset preservation 0..1. Above
 *   zero, transients in the input are detected (via a half-wave-rectified
 *   energy gradient) and phase randomisation is reduced locally around them so
 *   attacks survive the stretch instead of smearing into wash. `0` is classic
 *   PaulStretch (everything washed); `1` is strong onset preservation, good for
 *   percussive or plucked material.
 * @param {number} [options.spectralDrift=0] - How far the analysis read head
 *   wanders from its linear position, 0..1 (a fraction of the source;
 *   `1` ≈ ±30 %). `0` reads linearly, exactly as the classic algorithm does.
 *   Above zero the read position is offset by three sinusoids at golden-ratio
 *   rates, so the render keeps re-approaching the material from different
 *   places instead of walking it once at a constant rate. This is the single
 *   biggest lever on whether an hour-long render still feels alive after the
 *   first minute.
 * @param {number} [options.phaseContinuity=0] - How much of the previous
 *   window's phase carries into the next, 0..1. `0` is fully independent random
 *   phase — the classic PaulStretch grain. Around `0.3` keeps the drone
 *   character while taking the digital edge off. Only applies at full
 *   `phaseRandomness`, since partial randomisation is already anchored to the
 *   source's own phase.
 * @param {boolean} [options.wrapInput=false] - Read the input circularly
 *   instead of zero-padding past its ends. Used by seamless-loop renders:
 *   without it the analysis window progressively empties as the read head
 *   passes the input's end, so the final `windowSize / inputLen` fraction of
 *   the output decays to silence — glaring when the input is short relative to
 *   the window, and exactly the material a loop crossfade splices onto a
 *   full-energy head.
 * @param {bigint} [options.seed=DEFAULT_SEED] - Render seed. Phase
 *   randomisation is derived from it per window, so the same source + options
 *   + seed always produce bit-identical output.
 * @param {(frac: number) => void|Promise<void>} [options.onProgress] - Called
 *   every 64 windows with the completed fraction, and once with `1` at the
 *   end. Anything it returns is awaited, which is how a browser caller yields
 *   to the event loop (`() => new Promise(r => setTimeout(r, 0))`).
 * @returns {Promise<import('./buffer.js').StereoBuffer>} The stretched audio,
 *   peak-normalised to 0.92.
 *
 * @example
 * // Classic wash: ten seconds of field recording becomes eighty.
 * const washed = await stretch(source, 8);
 *
 * @example
 * // A long ambient bed that stays alive, reporting progress as it renders.
 * const bed = await stretch(source, 60, {
 *   windowSec: 0.35,
 *   spectralDrift: 0.4,
 *   phaseContinuity: 0.3,
 *   pitchSemitones: -7,
 *   seed: 0x1234abcdn,
 *   onProgress: (frac) => console.log(`${(frac * 100).toFixed(1)}%`),
 * });
 */
export async function stretch(input, ratio, options = {}) {
  const windowSec = options.windowSec ?? DEFAULT_WINDOW_SEC;
  const phaseRandomness = clamp01(options.phaseRandomness ?? 1);
  const onsetSensitivity = clamp01(options.onsetSensitivity ?? 0);
  const spectralDrift = clamp01(options.spectralDrift ?? 0);
  const phaseContinuity = clamp01(options.phaseContinuity ?? 0);
  const seed = options.seed ?? DEFAULT_SEED;
  const onProgress = options.onProgress;

  if (ratio <= 1.001) {
    // A copy, not the input itself: callers treat the result as theirs to
    // mutate (normalise, fade, splice), and handing back the source would let
    // a passthrough render quietly damage it.
    const copy = createBuffer(input.l.length, input.sampleRate);
    copy.l.set(input.l);
    copy.r.set(input.r);
    if (onProgress) { const p = onProgress(1); if (p) await p; }
    return copy;
  }

  const sr = input.sampleRate;

  // Pitch by RESAMPLING THE INPUT, not by remapping FFT bins.
  //
  // Bin remapping interpolated complex values linearly, which is not
  // phase-coherent: adjacent bins of a windowed sinusoid have a fixed phase
  // relationship and interpolation destroys it by an amount that depends on
  // where the fractional bin position falls. Shifting UP reads source
  // positions less than a bin apart, so consecutive output bins interpolated
  // repeatedly within the same pair and the error varied window to window —
  // audible as flutter, and measurably worse upward than downward.
  //
  // Resampling is what PaulStretch itself does. Reading the source 1/factor
  // times as fast raises its pitch by `factor`; the stretch ratio is scaled
  // to compensate, so the OUTPUT LENGTH is unchanged and every caller that
  // depends on it still gets what it asked for.
  const pitchSemitones = options.pitchSemitones ?? 0;
  const pitchFactor = Math.pow(2, pitchSemitones / 12);
  const shifting = Math.abs(pitchFactor - 1) > 0.001;
  const rawL = input.l;
  const rawR = input.r;
  let inL = rawL;
  let inR = rawR;
  let inputLen = rawL.length;
  if (shifting) {
    const by = 1 / pitchFactor;
    // Swift's `resampled(_:by:)` bails on sources too short to interpolate.
    if (inputLen > 3 && by > 0 && Math.abs(by - 1) > 0.0001) {
      const outCount = Math.max(4, Math.round(inputLen * by));
      inL = resampleChannel(rawL, inputLen, outCount, by);
      // A mono source shares one array between channels; resampling it twice
      // would produce the same samples at twice the cost.
      inR = rawR === rawL ? inL : resampleChannel(rawR, inputLen, outCount, by);
      inputLen = outCount;
    }
  }
  const effRatio = shifting ? ratio * pitchFactor : ratio;

  const wrapInput = (options.wrapInput ?? false) && inputLen > 0;

  const windowSize = nextPow2(Math.trunc(windowSec * sr));
  const halfWindow = windowSize >> 1;
  // 4× Hann overlap (hop = W/4). Applied TWICE — once on analysis and once on
  // synthesis. The SQUARED window must sum to a constant across the overlap
  // or the output is amplitude modulated at the hop rate no matter what the
  // spectral stage does. Hann² = 0.375 - 0.5cos(t) + 0.125cos(2t); at a hop
  // of N/4 the four overlapping copies put the cos(t) terms 90° apart and the
  // cos(2t) terms 180° apart, so both cancel and the sum is a flat 1.5.
  const outputStride = halfWindow >> 1;
  const inputStride = outputStride / effRatio;

  const outputLength = Math.max(windowSize, Math.trunc(inputLen * effRatio));
  const lastBlock = Math.trunc((outputLength - halfWindow - 1) / outputStride);
  const out = createBuffer(outputLength, sr);
  const outL = out.l;
  const outR = out.r;

  // SYMMETRIC Hann (denominator windowSize - 1): only the PERIODIC form
  // (/windowSize) sums exactly flat when squared at hop N/4; this one leaves
  // O(1/N) ripple (~-80 dB at N = 8192), inaudible and hidden by the
  // peak-normalise. Kept symmetric ON PURPOSE — switching to the periodic form
  // would change every sample of every seeded render and invalidate the parity
  // fingerprints for a difference nobody can hear.
  const hann = new Float32Array(windowSize);
  for (let i = 0; i < windowSize; i++) {
    hann[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (windowSize - 1));
  }

  const realL = new Float32Array(windowSize);
  const imagL = new Float32Array(windowSize);
  const realR = new Float32Array(windowSize);
  const imagR = new Float32Array(windowSize);

  // Onset curve — only computed when the user asks for it (cheap, but no
  // need to pay the cost when sensitivity is 0). Normalised 0..1; high
  // values mean "this input frame range is at or near a transient".
  //
  // Computed on the RESAMPLED source, since `inputStart` indexes that buffer
  // — a curve from the original input would place onset protection at times
  // off by the pitch factor.
  const onsetCurve = onsetSensitivity > 0
    ? computeOnsetCurve(inL, inR, ONSET_FRAME_SIZE)
    : null;

  /**
   * The analysis start frame for window `b`.
   *
   * Linear when `spectralDrift` is 0 (bit-identical to the classic
   * algorithm). Otherwise the linear position is offset by three sinusoids
   * whose rates are in golden-ratio proportion — chosen so the components are
   * mutually irrational and the combined motion has no audible period over an
   * hour-long render. Position stays a pure function of the window index, so
   * any range of the output can still be rendered independently.
   *
   * The two paths round differently on purpose: `Math.trunc` matches Swift's
   * `Int(_:)` on the (always non-negative) linear position, while the drift
   * path uses `Math.floor` to match `.rounded(.down)` — which differs from
   * truncation exactly when the offset drags the position negative.
   *
   * @param {number} b - Window index.
   * @returns {number} Start frame in the (possibly resampled) source. May be
   *   negative when drift is active.
   */
  const readPosition = (b) => {
    const linear = b * inputStride;
    if (!(spectralDrift > 0) || lastBlock <= 0 || inputLen <= 0) return Math.trunc(linear);
    const tau = b / lastBlock;                   // 0…1 through the render
    const w = (Math.sin(2 * Math.PI * PHI * tau)
             + Math.sin(2 * Math.PI * PHI * PHI * tau)
             + Math.sin(2 * Math.PI * PHI * PHI * PHI * tau)) / 3;
    // ±30 % of the source at full drift, matching the reference range.
    const offset = w * spectralDrift * 0.30 * inputLen;
    return Math.floor(linear + offset);
  };

  for (let b = 0; b <= lastBlock; b++) {
    const outputPos = b * outputStride;
    const inputStart = readPosition(b);

    if (wrapInput) {
      for (let i = 0; i < windowSize; i++) {
        // Floor-mod, not `%`: `readPosition` can return a NEGATIVE frame when
        // spectralDrift > 0 (the sinusoid offset can exceed the linear
        // position early in the render), and JS `%` preserves the dividend's
        // sign. Identical to `%` for the non-negative positions every
        // drift-free render uses.
        const idx = ((inputStart + i) % inputLen + inputLen) % inputLen;
        const w = hann[i];
        realL[i] = inL[idx] * w;
        realR[i] = inR[idx] * w;
        imagL[i] = 0;
        imagR[i] = 0;
      }
    } else {
      for (let i = 0; i < windowSize; i++) {
        const idx = inputStart + i;
        const w = hann[i];
        if (idx >= 0 && idx < inputLen) {
          realL[i] = inL[idx] * w;
          realR[i] = inR[idx] * w;
        } else {
          realL[i] = 0;
          realR[i] = 0;
        }
        imagL[i] = 0;
        imagR[i] = 0;
      }
    }

    fft(realL, imagL, false);
    fft(realR, imagR, false);
    // Modulate randomness by local onset strength. Near an attack, less
    // randomisation = transient phase relationships are preserved =
    // attack survives instead of smearing into the wash.
    let effRandomness = phaseRandomness;
    if (onsetCurve) {
      const frameIdx = Math.max(0, Math.min(onsetCurve.length - 1, Math.floor(inputStart / ONSET_FRAME_SIZE)));
      const localOnset = onsetCurve[frameIdx];
      effRandomness = phaseRandomness * (1 - onsetSensitivity * localOnset);
    }
    // One RNG per window, seeded from the window index, shared by L then R
    // — the exact consumption order the Swift kernel uses. Per-window
    // seeding is also what lets any chunk/worker reproduce a given window.
    const rng = new FastRNG(blockSeed(seed, b));
    // `prevRng` replays window b-1's stream in lockstep, so no state is
    // stored and any window can still be computed independently. Window 0 has
    // no predecessor, so it stays fully random.
    const prevRng = (phaseContinuity > 0 && b > 0) ? new FastRNG(blockSeed(seed, b - 1)) : null;
    randomizePhases(realL, imagL, windowSize, effRandomness, rng, prevRng, phaseContinuity);
    randomizePhases(realR, imagR, windowSize, effRandomness, rng, prevRng, phaseContinuity);
    fft(realL, imagL, true);
    fft(realR, imagR, true);

    const iEnd = Math.min(windowSize, outputLength - outputPos);
    for (let i = 0; i < iEnd; i++) {
      const w = hann[i];
      outL[outputPos + i] += realL[i] * w;
      outR[outputPos + i] += realR[i] * w;
    }

    if (onProgress && b % PROGRESS_BLOCKS === 0) {
      const p = onProgress(outputPos / outputLength);
      if (p) await p;
    }
  }

  normalizeToPeak(out, 0.92);

  if (onProgress) { const p = onProgress(1); if (p) await p; }
  return out;
}
