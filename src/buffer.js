/**
 * The neutral stereo buffer every engine in this package consumes and
 * produces.
 *
 * Deliberately NOT a Web Audio `AudioBuffer` — that type only exists in a
 * browser, and binding the DSP to it is what stops the same code running in
 * Node, a Worker, or a test harness. This mirrors `StereoBuffer` in the Swift
 * reference implementation, so the two ports stay structurally aligned.
 *
 * @typedef {object} StereoBuffer
 * @property {Float32Array} l Left channel samples.
 * @property {Float32Array} r Right channel samples. May be the SAME array as
 *   `l` for mono sources — engines treat that as a valid, cheap mono
 *   representation and never write through it.
 * @property {number} sampleRate Frames per second.
 */

/**
 * Allocate a silent stereo buffer.
 *
 * @param {number} frameCount - Length in frames. Negative values clamp to 0.
 * @param {number} sampleRate - Frames per second.
 * @returns {StereoBuffer} A buffer with two independent zeroed channels.
 *
 * @example
 * createBuffer(44100, 44100);  // one second of silence
 */
export function createBuffer(frameCount, sampleRate) {
  const n = Math.max(0, Math.trunc(frameCount));
  return { l: new Float32Array(n), r: new Float32Array(n), sampleRate };
}

/**
 * Build a stereo buffer from existing channel data.
 *
 * Passing only `l` produces a mono buffer whose two channels share one array,
 * which every engine handles without copying.
 *
 * @param {Float32Array} l - Left channel.
 * @param {Float32Array} [r] - Right channel. Defaults to `l` (mono).
 * @param {number} [sampleRate] - Frames per second. Default 44100.
 * @returns {StereoBuffer}
 *
 * @example
 * fromChannels(samples, undefined, 48000);  // mono at 48 kHz
 * fromChannels(left, right, 44100);         // true stereo
 */
export function fromChannels(l, r, sampleRate = 44100) {
  return { l, r: r ?? l, sampleRate };
}

/**
 * Frame count of a buffer (its left channel's length).
 *
 * @param {StereoBuffer} buffer
 * @returns {number} Length in frames.
 *
 * @example
 * frameCount(createBuffer(1024, 44100));  // 1024
 */
export function frameCount(buffer) {
  return buffer.l.length;
}

/**
 * Duration of a buffer in seconds.
 *
 * @param {StereoBuffer} buffer
 * @returns {number} Seconds. Returns 0 when the sample rate is not positive.
 *
 * @example
 * duration(createBuffer(22050, 44100));  // 0.5
 */
export function duration(buffer) {
  return buffer.sampleRate > 0 ? buffer.l.length / buffer.sampleRate : 0;
}

/**
 * Largest absolute sample value across both channels.
 *
 * @param {StereoBuffer} buffer
 * @returns {number} Peak magnitude. 0 for a silent or empty buffer.
 *
 * @example
 * peak(fromChannels(new Float32Array([0.1, -0.7, 0.3])));  // 0.7
 */
export function peak(buffer) {
  let max = 0;
  const { l, r } = buffer;
  for (let i = 0; i < l.length; i++) {
    const a = Math.abs(l[i]);
    if (a > max) max = a;
  }
  if (r !== l) {
    for (let i = 0; i < r.length; i++) {
      const b = Math.abs(r[i]);
      if (b > max) max = b;
    }
  }
  return max;
}

/**
 * Scale both channels in place by ONE factor so the louder of the two peaks at
 * `target`.
 *
 * A single shared gain is the point: normalising the channels independently
 * would collapse the stereo image of anything panned off centre. Silent
 * buffers are left untouched rather than divided by zero.
 *
 * @param {StereoBuffer} buffer - Mutated in place.
 * @param {number} [target] - Desired peak magnitude. Default 0.92, matching
 *   the Swift reference's headroom.
 * @returns {StereoBuffer} The same buffer, for chaining.
 *
 * @example
 * normalizeToPeak(rendered);        // peak lands on 0.92
 * normalizeToPeak(rendered, 1.0);   // peak lands on full scale
 */
export function normalizeToPeak(buffer, target = 0.92) {
  const p = peak(buffer);
  if (p <= 0) return buffer;
  const gain = target / p;
  const { l, r } = buffer;
  for (let i = 0; i < l.length; i++) l[i] *= gain;
  if (r !== l) for (let i = 0; i < r.length; i++) r[i] *= gain;
  return buffer;
}
