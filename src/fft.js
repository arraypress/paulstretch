/**
 * In-place Cooley-Tukey radix-2 FFT.
 *
 * Kept hand-rolled rather than pulling in a generic DSP library: at the window
 * sizes these engines use (8192 frames is typical) the per-window phase work
 * dominates the cost, not the transform, so a dependency would buy nothing and
 * cost portability.
 *
 * Convention matches the Swift reference: the forward transform is
 * **unscaled**, the inverse is normalised by `1/N`, so a forward → inverse
 * round-trip is the identity.
 */

/**
 * Transform `real` / `imag` in place.
 *
 * Both arrays are mutated; output replaces input.
 *
 * @param {Float32Array} real - Real components. Length MUST be a power of two —
 *   the bit-reversal loop silently produces garbage otherwise.
 * @param {Float32Array} imag - Imaginary components, same length as `real`.
 * @param {boolean} inverse - `true` for the inverse transform (which also
 *   applies the `1/N` normalisation).
 * @returns {void}
 *
 * @example
 * const re = new Float32Array([1, 0, 0, 0]);
 * const im = new Float32Array(4);
 * fft(re, im, false);   // forward — a unit impulse becomes a flat spectrum
 * fft(re, im, true);    // inverse — back to the original impulse
 */
export function fft(real, imag, inverse) {
  const n = real.length;

  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = real[i];
      real[i] = real[j];
      real[j] = t;
      t = imag[i];
      imag[i] = imag[j];
      imag[j] = t;
    }
  }

  for (let len = 2; len <= n; len <<= 1) {
    const halfLen = len >> 1;
    const angle = ((inverse ? 2 : -2) * Math.PI) / len;
    const wReal = Math.cos(angle);
    const wImag = Math.sin(angle);
    for (let i = 0; i < n; i += len) {
      let curReal = 1;
      let curImag = 0;
      for (let j = 0; j < halfLen; j++) {
        const aReal = real[i + j];
        const aImag = imag[i + j];
        const bReal = real[i + j + halfLen] * curReal - imag[i + j + halfLen] * curImag;
        const bImag = real[i + j + halfLen] * curImag + imag[i + j + halfLen] * curReal;
        real[i + j] = aReal + bReal;
        imag[i + j] = aImag + bImag;
        real[i + j + halfLen] = aReal - bReal;
        imag[i + j + halfLen] = aImag - bImag;
        const nextReal = curReal * wReal - curImag * wImag;
        const nextImag = curReal * wImag + curImag * wReal;
        curReal = nextReal;
        curImag = nextImag;
      }
    }
  }

  if (inverse) {
    const invN = 1 / n;
    for (let i = 0; i < n; i++) {
      real[i] *= invN;
      imag[i] *= invN;
    }
  }
}
