/**
 * FFT correctness, checked against a naive DFT rather than against itself.
 *
 * The convention under test — unscaled forward, 1/N inverse — is the one the
 * Swift reference uses. Get it wrong and every render comes out at a different
 * LEVEL while still looking structurally correct, which is a miserable bug to
 * chase from the audio alone.
 */
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { fft } from '../src/fft.js';

/** Textbook O(n^2) DFT, used only as an independent oracle. */
function naiveDft(real, imag, inverse) {
  const n = real.length;
  const outRe = new Float64Array(n);
  const outIm = new Float64Array(n);
  const sign = inverse ? 2 : -2;
  for (let k = 0; k < n; k++) {
    let sre = 0;
    let sim = 0;
    for (let t = 0; t < n; t++) {
      const angle = (sign * Math.PI * k * t) / n;
      const c = Math.cos(angle);
      const s = Math.sin(angle);
      sre += real[t] * c - imag[t] * s;
      sim += real[t] * s + imag[t] * c;
    }
    outRe[k] = inverse ? sre / n : sre;
    outIm[k] = inverse ? sim / n : sim;
  }
  return { re: outRe, im: outIm };
}

function assertClose(got, want, tol, label) {
  for (let i = 0; i < want.length; i++) {
    assert.ok(
      Math.abs(got[i] - want[i]) < tol,
      `${label}[${i}]: got ${got[i]}, want ${want[i]}`,
    );
  }
}

describe('fft', () => {
  it('matches a naive DFT on the forward transform', () => {
    const n = 64;
    const re = new Float32Array(n);
    const im = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      re[i] = Math.sin((2 * Math.PI * 3 * i) / n) + 0.4 * Math.cos((2 * Math.PI * 11 * i) / n);
    }
    const want = naiveDft(re, im, false);
    fft(re, im, false);
    assertClose(re, want.re, 1e-3, 'real');
    assertClose(im, want.im, 1e-3, 'imag');
  });

  it('round-trips forward then inverse to the identity', () => {
    const n = 256;
    const re = new Float32Array(n);
    const im = new Float32Array(n);
    const original = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      original[i] = Math.sin(i * 0.37) * 0.5 + Math.sin(i * 1.9) * 0.25;
      re[i] = original[i];
    }
    fft(re, im, false);
    fft(re, im, true);
    assertClose(re, original, 1e-5, 'round-trip real');
    for (let i = 0; i < n; i++) assert.ok(Math.abs(im[i]) < 1e-5, `imag leaked at ${i}`);
  });

  it('leaves the forward transform UNSCALED', () => {
    // Bin 0 of a forward transform is the plain sum of the input. If a stray
    // 1/N crept into the forward direction this is what would catch it.
    const n = 32;
    const re = new Float32Array(n).fill(1);
    const im = new Float32Array(n);
    fft(re, im, false);
    assert.ok(Math.abs(re[0] - n) < 1e-3, `bin 0 was ${re[0]}, expected ${n}`);
  });

  it('normalises the inverse by 1/N', () => {
    const n = 32;
    const re = new Float32Array(n);
    const im = new Float32Array(n);
    re[0] = n; // flat spectrum in bin 0
    fft(re, im, true);
    for (let i = 0; i < n; i++) {
      assert.ok(Math.abs(re[i] - 1) < 1e-5, `sample ${i} was ${re[i]}, expected 1`);
    }
  });

  it('turns a unit impulse into a flat spectrum', () => {
    const n = 16;
    const re = new Float32Array(n);
    const im = new Float32Array(n);
    re[0] = 1;
    fft(re, im, false);
    for (let i = 0; i < n; i++) {
      assert.ok(Math.abs(re[i] - 1) < 1e-6, `real bin ${i} was ${re[i]}`);
      assert.ok(Math.abs(im[i]) < 1e-6, `imag bin ${i} was ${im[i]}`);
    }
  });

  it('puts a pure tone in exactly one conjugate bin pair', () => {
    const n = 64;
    const k = 5;
    const re = new Float32Array(n);
    const im = new Float32Array(n);
    for (let i = 0; i < n; i++) re[i] = Math.cos((2 * Math.PI * k * i) / n);
    fft(re, im, false);
    const mag = (i) => Math.hypot(re[i], im[i]);
    assert.ok(Math.abs(mag(k) - n / 2) < 1e-2, `bin ${k} magnitude was ${mag(k)}`);
    assert.ok(Math.abs(mag(n - k) - n / 2) < 1e-2, `mirror bin magnitude was ${mag(n - k)}`);
    for (let i = 0; i < n; i++) {
      if (i === k || i === n - k) continue;
      assert.ok(mag(i) < 1e-2, `bin ${i} should be empty, was ${mag(i)}`);
    }
  });

  it('handles the degenerate length 1', () => {
    const re = new Float32Array([0.5]);
    const im = new Float32Array([0]);
    fft(re, im, false);
    assert.equal(re[0], 0.5);
  });

  it('operates in place', () => {
    const re = new Float32Array([1, 2, 3, 4]);
    const im = new Float32Array(4);
    const ref = re;
    fft(re, im, false);
    assert.equal(ref, re, 'array identity changed — transform was not in place');
  });
});
