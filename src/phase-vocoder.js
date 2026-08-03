/**
 * Phase-vocoder time stretch — a port of `PhaseVocoder.swift`
 * (swift-paul-stretch).
 *
 * Unlike PaulStretch, phases are *propagated* between windows from a per-bin
 * instantaneous-frequency estimate rather than randomised, so the source's
 * structure survives: pitch, transients and movement stay recognisable
 * instead of dissolving into wash. That makes it the engine to reach for when
 * you want the material stretched but still *itself*.
 *
 * Phase accumulation is inherently sequential — window b's output phase
 * depends on b-1 — so unlike the other engines this cannot be rendered as
 * independent ranges, and there is no seed: nothing here is random. The Swift
 * version wraps the sequencing in a stateful stream; here the whole timeline
 * is produced in one pass, which is equivalent and simpler (the
 * accumulator/shift dance over there exists only to serve ranged emission).
 */

import { fft } from './fft.js';
import { nextPow2 } from './rng.js';
import { createBuffer } from './buffer.js';

/**
 * @typedef {object} PhaseVocoderOptions
 * @property {number} [windowSec=0.25] STFT window in seconds. Rounded up to a
 *   power of two frames internally. Longer windows smear transients but track
 *   low frequencies better; 0.25 s is the reference default.
 * @property {number} [pitchSemitones=0] Pitch shift in semitones, applied by
 *   spectral bin remapping. Shifts smaller than ~0.017 semitones are treated
 *   as no shift at all and skip the remap entirely.
 * @property {boolean} [wrapInput=false] Read the input circularly instead of
 *   zero-padding past its ends, so a seamless-loop render never fades into
 *   silence at the seams.
 * @property {number} [outputFrames] Deliver exactly this many frames instead
 *   of the engine's natural output length, zero-padding or stopping short as
 *   needed.
 * @property {(frac: number) => (void | Promise<void>)} [onProgress] Called
 *   every 16 windows with a 0..1 fraction, and once with 1 at the end. If it
 *   returns a promise it is awaited, which is how a caller yields to an event
 *   loop mid-render. Omit it and nothing is called or awaited.
 */

/**
 * Wraps a phase to (-pi, pi].
 *
 * `%` on a negative left operand truncates toward zero in JavaScript, exactly
 * like Swift's `truncatingRemainder`, so the two branches below see the same
 * input on both sides of the port.
 */
function princarg(x) {
  let a = x % (2 * Math.PI);
  if (a > Math.PI) a -= 2 * Math.PI;
  if (a <= -Math.PI) a += 2 * Math.PI;
  return a;
}

/**
 * Spectral bin remap for pitch: output bin k reads input bin k/factor.
 *
 * Bins whose source falls outside the real spectrum are zeroed rather than
 * clamped — clamping would pile the whole out-of-range tail onto one bin and
 * ring. The result is written through `tmpR`/`tmpI` because the read indices
 * (`lo`, `hi`) run behind and ahead of `k` for shifts either way, so remapping
 * in place would consume bins it had already overwritten.
 */
function pitchShiftBins(real, imag, n, factor, tmpR, tmpI) {
  const half = n >> 1;
  tmpR[0] = real[0]; tmpI[0] = 0;
  for (let k = 1; k < half; k++) {
    const sourceK = k / factor;
    const lo = Math.floor(sourceK);
    const hi = lo + 1;
    if (lo < 1 || hi >= half) {
      tmpR[k] = 0; tmpI[k] = 0;
    } else {
      const frac = sourceK - lo;
      tmpR[k] = real[lo] * (1 - frac) + real[hi] * frac;
      tmpI[k] = imag[lo] * (1 - frac) + imag[hi] * frac;
    }
    // Mirror the bin into the upper half as its conjugate, so the inverse
    // transform yields a real signal.
    tmpR[n - k] = tmpR[k];
    tmpI[n - k] = -tmpI[k];
  }
  tmpR[half] = 0; tmpI[half] = 0;
  real.set(tmpR); imag.set(tmpI);
}

/**
 * Stretch a buffer by `ratio` with phase propagation.
 *
 * The output is NOT normalised. That is deliberate and it is the one thing
 * callers must remember: the Swift renderer derives this engine's output gain
 * from a sweep-approximated peak, so an exact normalise here would not
 * reproduce its level anyway, and the raw 4x-Hann overlap-add sits around 1.5x
 * the source peak — well past full scale. Run `normalizeToPeak(out, 0.92)` on
 * the result unless you have your own gain staging.
 *
 * `ratio` is output length over input length: 8 makes a 30-second source four
 * minutes long, 0.5 halves it. Values below 1 are fine; the analysis hop just
 * advances faster than the synthesis hop.
 *
 * @param {import('./buffer.js').StereoBuffer} input - Source audio. A mono
 *   buffer (both channels sharing one array) is processed as two identical
 *   channels.
 * @param {number} ratio - Time-stretch factor, output frames per input frame.
 * @param {PhaseVocoderOptions} [options] - Window, pitch and delivery options.
 * @returns {Promise<import('./buffer.js').StereoBuffer>} A new buffer at the
 *   input's sample rate, `outputFrames` long if given, otherwise
 *   `max(windowFrames, trunc(inputFrames * ratio))`. An empty input yields
 *   that many frames of silence rather than an empty buffer.
 *
 * @example
 * import { phaseVocoder, normalizeToPeak } from '@arraypress/paulstretch';
 *
 * // A 30 s field recording stretched to four minutes, still recognisable.
 * const out = await phaseVocoder(source, 8);
 * normalizeToPeak(out, 0.92);
 *
 * @example
 * // Seamless loop, dropped a fifth, delivered at exactly the length the
 * // caller already allocated for — and yielding to the event loop as it goes.
 * const out = await phaseVocoder(source, 12, {
 *   windowSec: 0.25,
 *   pitchSemitones: -7,
 *   wrapInput: true,
 *   outputFrames: Math.floor(600 * source.sampleRate),
 *   onProgress: (f) => new Promise((r) => setTimeout(() => { report(f); r(); }, 0)),
 * });
 */
export async function phaseVocoder(input, ratio, options = {}) {
  const sr = input.sampleRate;
  const inL = input.l;
  const inR = input.r;
  const inputLen = inL.length;

  const windowSec = options.windowSec ?? 0.25;
  const pitchSemitones = options.pitchSemitones ?? 0;
  const onProgress = options.onProgress;
  const pitchFactor = Math.pow(2, pitchSemitones / 12);
  const doPitch = Math.abs(pitchFactor - 1) > 0.001;
  const wrapInput = (options.wrapInput ?? false) && inputLen > 0;

  const W = nextPow2(Math.trunc(windowSec * sr));
  const half = W >> 1;
  const hop = (W >> 1) >> 1;                // synthesis hop, W/4
  const inputStride = hop / ratio;          // analysis hop (fractional)
  const outputLength = Math.max(W, Math.trunc(inputLen * ratio));
  const lastBlock = Math.trunc((outputLength - half - 1) / hop);

  // The window schedule stays driven by `outputLength`; only how much of it
  // we hand back is `deliverLength`. Swift keeps these separate too: the STFT
  // schedule follows `PVSpec.outputLength` while `StretchRenderer` asks the
  // stream for exactly `preLoopFrames` and the stream emits silence past the
  // end. The two differ by a sample or so on most source/target pairs because
  // `trunc(inputLen * ratio)` and `floor(targetSec * sr)` round differently,
  // and materialising a whole second buffer to reconcile that is a real cost
  // at 30-plus-minute lengths.
  const deliverLength = Math.max(1, Math.trunc(options.outputFrames ?? outputLength));
  const out = createBuffer(deliverLength, sr);
  const outL = out.l;
  const outR = out.r;
  if (inputLen <= 0) {
    if (onProgress) { const p = onProgress(1); if (p) await p; }
    return out;
  }

  // SYMMETRIC Hann (denominator W - 1): only the PERIODIC form (/W) sums
  // exactly flat when squared at hop W/4; this one leaves O(1/N) ripple
  // (~-80 dB at N = 8192), inaudible and hidden by the caller's normalise.
  // Kept symmetric because changing it would invalidate every render this
  // port has been verified against.
  const hann = new Float32Array(W);
  for (let i = 0; i < W; i++) hann[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (W - 1));

  const realL = new Float32Array(W), imagL = new Float32Array(W);
  const realR = new Float32Array(W), imagR = new Float32Array(W);
  const tmpR = new Float32Array(W), tmpI = new Float32Array(W);
  // Phases stay in float64: they accumulate across every window of the render,
  // so float32 rounding would compound into audible drift.
  const prevPhaseL = new Float64Array(half + 1), prevPhaseR = new Float64Array(half + 1);
  const outPhaseL = new Float64Array(half + 1), outPhaseR = new Float64Array(half + 1);

  let prevInputStart = 0;

  for (let b = 0; b <= lastBlock; b++) {
    const inputStart = Math.trunc(b * inputStride);

    for (let i = 0; i < W; i++) {
      const w = hann[i];
      if (wrapInput) {
        const idx = (inputStart + i) % inputLen;
        realL[i] = inL[idx] * w; realR[i] = inR[idx] * w;
      } else {
        const idx = inputStart + i;
        if (idx >= 0 && idx < inputLen) { realL[i] = inL[idx] * w; realR[i] = inR[idx] * w; }
        else { realL[i] = 0; realR[i] = 0; }
      }
      imagL[i] = 0; imagR[i] = 0;
    }
    fft(realL, imagL, false);
    fft(realR, imagR, false);

    // The analysis hop actually advanced (integer, and 0 at extreme ratios —
    // then only the expected phase advance applies).
    const haActual = inputStart - prevInputStart;
    const first = b === 0;

    for (let k = 0; k <= half; k++) {
      const omega = (2 * Math.PI * k) / W;           // rad/sample
      const expectedOut = omega * hop;               // per synthesis hop

      const reL = realL[k], imL = imagL[k];
      const magL = Math.sqrt(reL * reL + imL * imL);
      const phL = Math.atan2(imL, reL);
      const reR = realR[k], imR = imagR[k];
      const magR = Math.sqrt(reR * reR + imR * imR);
      const phR = Math.atan2(imR, reR);

      if (first) {
        // Nothing to propagate from: adopt the analysis phase so the very
        // first window resynthesises the source exactly.
        outPhaseL[k] = phL;
        outPhaseR[k] = phR;
      } else if (haActual > 0) {
        // Heterodyne: strip the phase advance the bin centre frequency would
        // have produced, wrap the remainder to get the true deviation, then
        // replay expected + deviation over the SYNTHESIS hop instead.
        const expectedIn = omega * haActual;
        const dL = princarg(phL - prevPhaseL[k] - expectedIn);
        const dR = princarg(phR - prevPhaseR[k] - expectedIn);
        const scale = hop / haActual;
        outPhaseL[k] = princarg(outPhaseL[k] + (expectedIn + dL) * scale);
        outPhaseR[k] = princarg(outPhaseR[k] + (expectedIn + dR) * scale);
      } else {
        // Stretched so far that consecutive windows read the same input
        // frame: there is no measurable deviation, so advance at the bin's
        // own frequency and let the magnitudes carry the movement.
        outPhaseL[k] = princarg(outPhaseL[k] + expectedOut);
        outPhaseR[k] = princarg(outPhaseR[k] + expectedOut);
      }
      prevPhaseL[k] = phL;
      prevPhaseR[k] = phR;

      // Resynthesise the bin, conjugate-mirrored for a real inverse
      // transform. DC and Nyquist have no partner to mirror into and must
      // stay purely real, hence the special case.
      const rvL = magL * Math.cos(outPhaseL[k]), ivL = magL * Math.sin(outPhaseL[k]);
      const rvR = magR * Math.cos(outPhaseR[k]), ivR = magR * Math.sin(outPhaseR[k]);
      if (k === 0 || k === half) {
        realL[k] = rvL; imagL[k] = 0;
        realR[k] = rvR; imagR[k] = 0;
      } else {
        realL[k] = rvL; imagL[k] = ivL;
        realL[W - k] = rvL; imagL[W - k] = -ivL;
        realR[k] = rvR; imagR[k] = ivR;
        realR[W - k] = rvR; imagR[W - k] = -ivR;
      }
    }
    prevInputStart = inputStart;

    if (doPitch) {
      pitchShiftBins(realL, imagL, W, pitchFactor, tmpR, tmpI);
      pitchShiftBins(realR, imagR, W, pitchFactor, tmpR, tmpI);
    }

    fft(realL, imagL, true);
    fft(realR, imagR, true);

    // Overlap-add straight into the timeline: window b spans [b*hop, +W).
    // Later windows start later still, so once one clears the delivered
    // range there is nothing left to write.
    const base = b * hop;
    if (base >= deliverLength) break;
    const iEnd = Math.min(W, deliverLength - base);
    for (let i = 0; i < iEnd; i++) {
      const w = hann[i];
      outL[base + i] += realL[i] * w;
      outR[base + i] += realR[i] * w;
    }

    if ((b & 15) === 0 && onProgress) {
      const p = onProgress(b / (lastBlock + 1));
      if (p) await p;
    }
  }

  if (onProgress) { const p = onProgress(1); if (p) await p; }
  return out;
}
