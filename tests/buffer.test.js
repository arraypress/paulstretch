/**
 * The neutral buffer type every engine consumes.
 *
 * The mono case gets particular attention: `fromChannels(l)` deliberately makes
 * `r` the SAME array as `l` rather than copying, so anything that writes
 * through one channel would corrupt the other. Several tests below exist purely
 * to pin that contract down.
 */
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  createBuffer,
  fromChannels,
  frameCount,
  duration,
  peak,
  normalizeToPeak,
} from '../src/buffer.js';

describe('createBuffer', () => {
  it('allocates two independent silent channels', () => {
    const b = createBuffer(8, 44100);
    assert.equal(b.l.length, 8);
    assert.equal(b.r.length, 8);
    assert.notEqual(b.l, b.r, 'channels must not be aliased');
    assert.ok(b.l.every((v) => v === 0));
    assert.equal(b.sampleRate, 44100);
  });

  it('truncates a fractional frame count', () => {
    assert.equal(createBuffer(10.9, 44100).l.length, 10);
  });

  it('clamps a negative frame count to zero', () => {
    assert.equal(createBuffer(-5, 44100).l.length, 0);
  });
});

describe('fromChannels', () => {
  it('shares one array for mono', () => {
    const l = new Float32Array([1, 2, 3]);
    const b = fromChannels(l);
    assert.equal(b.l, b.r, 'mono should alias, not copy');
    assert.equal(b.sampleRate, 44100);
  });

  it('keeps both channels for stereo', () => {
    const l = new Float32Array([1, 2]);
    const r = new Float32Array([3, 4]);
    const b = fromChannels(l, r, 48000);
    assert.equal(b.l, l);
    assert.equal(b.r, r);
    assert.equal(b.sampleRate, 48000);
  });

  it('treats an explicit undefined right channel as mono', () => {
    const l = new Float32Array([1]);
    assert.equal(fromChannels(l, undefined, 22050).r, l);
  });
});

describe('frameCount / duration', () => {
  it('reports the left channel length', () => {
    assert.equal(frameCount(createBuffer(1024, 44100)), 1024);
  });

  it('converts frames to seconds', () => {
    assert.equal(duration(createBuffer(22050, 44100)), 0.5);
    assert.equal(duration(createBuffer(48000, 48000)), 1);
  });

  it('returns zero duration for a non-positive sample rate', () => {
    assert.equal(duration(fromChannels(new Float32Array(100), undefined, 0)), 0);
  });

  it('handles an empty buffer', () => {
    const b = createBuffer(0, 44100);
    assert.equal(frameCount(b), 0);
    assert.equal(duration(b), 0);
  });
});

describe('peak', () => {
  it('finds the largest magnitude across both channels', () => {
    // Compared with a tolerance throughout: Float32Array storage rounds, so
    // 0.9 reads back as 0.8999999761581421.
    const b = fromChannels(new Float32Array([0.1, -0.7]), new Float32Array([0.2, 0.9]));
    assert.ok(Math.abs(peak(b) - 0.9) < 1e-7);
  });

  it('respects a negative peak', () => {
    const b = fromChannels(new Float32Array([0.1, -0.8]), new Float32Array([0.2, 0.3]));
    assert.ok(Math.abs(peak(b) - 0.8) < 1e-7);
  });

  it('returns 0 for silence and for an empty buffer', () => {
    assert.equal(peak(createBuffer(16, 44100)), 0);
    assert.equal(peak(createBuffer(0, 44100)), 0);
  });

  it('scans a mono buffer once without missing anything', () => {
    const b = fromChannels(new Float32Array([0.25, -0.5, 0.1]));
    assert.ok(Math.abs(peak(b) - 0.5) < 1e-7);
  });
});

describe('normalizeToPeak', () => {
  it('scales the louder channel to the target', () => {
    const b = fromChannels(new Float32Array([0.1, 0.2]), new Float32Array([0.4, 0]));
    normalizeToPeak(b, 0.92);
    assert.ok(Math.abs(peak(b) - 0.92) < 1e-6);
  });

  it('defaults to 0.92', () => {
    const b = fromChannels(new Float32Array([0.5]), new Float32Array([0.25]));
    normalizeToPeak(b);
    assert.ok(Math.abs(b.l[0] - 0.92) < 1e-6);
  });

  it('applies ONE shared gain so the stereo image survives', () => {
    // Normalising per channel would push the quiet side up and centre the
    // image — the thing this function exists to avoid.
    const b = fromChannels(new Float32Array([0.4]), new Float32Array([0.2]));
    normalizeToPeak(b, 0.8);
    assert.ok(Math.abs(b.l[0] - 0.8) < 1e-6);
    assert.ok(Math.abs(b.r[0] - 0.4) < 1e-6, 'ratio between channels changed');
  });

  it('leaves silence untouched instead of dividing by zero', () => {
    const b = createBuffer(4, 44100);
    normalizeToPeak(b);
    assert.ok(b.l.every((v) => v === 0));
    assert.ok(b.r.every((v) => Number.isFinite(v)));
  });

  it('does not double-scale a mono buffer', () => {
    const l = new Float32Array([0.5, -0.25]);
    const b = fromChannels(l);
    normalizeToPeak(b, 1);
    assert.ok(Math.abs(b.l[0] - 1) < 1e-6, `got ${b.l[0]} — aliased channel scaled twice?`);
  });

  it('returns the same buffer for chaining', () => {
    const b = createBuffer(2, 44100);
    assert.equal(normalizeToPeak(b), b);
  });
});
