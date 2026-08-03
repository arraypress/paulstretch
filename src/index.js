/**
 * @arraypress/paulstretch
 *
 * Extreme time-stretching and ambient texture synthesis: four engines that turn
 * a few seconds of source into minutes or hours of evolving material.
 *
 * Every engine is buffer-in / buffer-out over a plain `{ l, r, sampleRate }`
 * object, so nothing here needs a browser, a Web Audio context, or a DOM. The
 * same code runs in Node, a Worker, a Cloudflare Worker, or a page.
 *
 * Output is fully deterministic — the same source, options and seed always
 * produce bit-identical audio. That is not decoration: it is what lets a render
 * be split across workers and reassembled, and it is what makes this port
 * verifiable against the Swift reference implementation it was derived from
 * (see the parity fingerprints in `tests/`).
 */

export {
  createBuffer,
  fromChannels,
  frameCount,
  duration,
  peak,
  normalizeToPeak,
} from './buffer.js';

export { fft } from './fft.js';

export { FastRNG, mixSeed, blockSeed, nextPow2 } from './rng.js';

export { DEFAULT_SEED, stretch } from './stretch.js';
export { freeze } from './freeze.js';
export { granular } from './granular.js';
export { phaseVocoder } from './phase-vocoder.js';

export { LAYER_PRESETS, pickLayers, layerSeed, layerPitch } from './layers.js';
