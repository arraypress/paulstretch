# Changelog

## 1.0.0

Initial release. Extracted from the Driftscape web app, where the DSP was
coupled to the Web Audio API, and verified against the Swift reference
implementation it was originally ported from.

### Engines

- `stretch` — PaulStretch phase-randomised time stretch, with `spectralDrift`,
  `phaseContinuity`, `onsetSensitivity`, `wrapInput` and resample-based pitch.
- `freeze` — spectral freeze with magnitude smear and a scanning capture point.
- `granular` — granular cloud with position, time, pitch and pan jitter.
- `phaseVocoder` — phase-propagating stretch, with an `outputFrames` option so a
  caller never has to allocate a second buffer to reconcile rounding.

### Layered drones

- `LAYER_PRESETS`, `pickLayers`, `layerSeed`, `layerPitch` — the six recipes from
  the Swift reference, including the two shimmer variants, plus the seed
  decorrelation and pitch stacking a caller needs to render them.

### Buffers and primitives

- `StereoBuffer` (`{ l, r, sampleRate }`) replaces `AudioBuffer` throughout, so
  the package has no browser dependency.
- `createBuffer`, `fromChannels`, `frameCount`, `duration`, `peak`,
  `normalizeToPeak`.
- `fft`, `FastRNG`, `mixSeed`, `blockSeed`, `nextPow2`, `DEFAULT_SEED`.

### Notes

- Engines are `async` and take an optional `onProgress` hook. Returning a promise
  from it makes the engine await, which is how a browser caller yields mid-render;
  omitting it means nothing is called or awaited.
- Output is bit-identical for a given source, options and seed. The test suite
  pins RMS fingerprints captured from real Swift renders.
