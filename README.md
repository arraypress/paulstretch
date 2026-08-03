# @arraypress/paulstretch

Extreme time-stretching and ambient texture synthesis -- PaulStretch, spectral freeze, granular cloud, and phase vocoder. Zero dependencies.

Works in Node.js, Cloudflare Workers, Deno, Bun, and browsers.

Turns a few seconds of source into minutes or hours of evolving material. Every engine is buffer-in / buffer-out over a plain object, so nothing here needs a Web Audio context or a DOM.

Output is deterministic: the same source, options and seed always produce bit-identical audio. That is what lets a render be split across workers and reassembled, and it is what makes this package verifiable against the [Swift reference implementation](https://github.com/arraypress/swift-paul-stretch) it was derived from. The test suite pins RMS fingerprints captured from real Swift renders.

## Install

```bash
npm install @arraypress/paulstretch
```

## The buffer type

Everything takes and returns a `StereoBuffer` -- a plain `{ l, r, sampleRate }` object, deliberately not a Web Audio `AudioBuffer`:

```js
import { fromChannels, createBuffer } from '@arraypress/paulstretch';

fromChannels(left, right, 44100)   // stereo
fromChannels(samples, undefined, 44100)  // mono: r aliases l, no copy
createBuffer(44100, 44100)         // one second of silence
```

In a browser, bridging to Web Audio is two lines each way:

```js
const input = fromChannels(
  audioBuffer.getChannelData(0),
  audioBuffer.numberOfChannels > 1 ? audioBuffer.getChannelData(1) : undefined,
  audioBuffer.sampleRate,
);

const out = ctx.createBuffer(2, result.l.length, result.sampleRate);
out.copyToChannel(result.l, 0);
out.copyToChannel(result.r, 1);
```

## Engines

### `stretch(input, ratio, options?)`

PaulStretch. Windows the source, replaces each window's phases with random ones, and overlap-adds at a quarter-window hop. Because the phases are random, windows can be pulled from the input far slower than they are laid down on the output -- stretching by enormous ratios with no pitch change and no time-locked transients. The result is the characteristic smeared, choir-like wash.

Returns audio ~`ratio` times longer, peak-normalised to 0.92. A ratio of 1.001 or below returns an unmodified copy.

```js
import { stretch, fromChannels } from '@arraypress/paulstretch';

const washed = await stretch(source, 40);

const alive = await stretch(source, 200, {
  windowSec: 0.4,
  spectralDrift: 0.5,     // stop the read head walking the source once, linearly
  phaseContinuity: 0.3,   // take the digital edge off the grain
});
```

| Option | Default | |
|---|---|---|
| `windowSec` | `0.25` | STFT window. Bigger = washier and more diffuse. |
| `phaseRandomness` | `1` | `1` is the classic wash; `0` leaves phase untouched. |
| `pitchSemitones` | `0` | Applied by resampling the source. Output length is preserved. |
| `onsetSensitivity` | `0` | How strongly rising-energy moments ease off the scramble. |
| `spectralDrift` | `0` | How far the analysis read head wanders from linear, 0..1. |
| `phaseContinuity` | `0` | How much of the previous window's phase carries over. |
| `wrapInput` | `false` | Read the source circularly, for seamless loops. |
| `seed` | `DEFAULT_SEED` | |
| `onProgress` | -- | Called every 64 windows, then once with `1`. |

**`spectralDrift`** is the single biggest lever on whether an hour-long render still feels alive. At `0` the read head walks the source once at a constant rate, so after a minute nothing new is happening. Above zero it is offset by three sinusoids at golden-ratio-related rates -- mutually irrational, so the motion never audibly repeats -- and the render keeps re-approaching the material from different places.

**`wrapInput`** matters for loops. Without it the analysis window progressively empties as the read head passes the source's end, so the final stretch of output decays toward silence -- which is exactly the material a loop crossfade splices onto a full-energy head.

### `freeze(input, targetSeconds, options?)`

Spectral freeze. Captures the magnitude spectrum at a single instant and resynthesises it forever with fresh random phase every hop: one frozen moment, shimmering indefinitely. The most suspended of the engines -- nothing of the source's movement survives, which is the point.

Peak-normalised to 0.92. Sources under 32 frames yield silence.

```js
import { freeze } from '@arraypress/paulstretch';

const pad = await freeze(source, 600, {
  positionNorm: 0.3,  // capture 30% of the way in
  smear: 0.2,         // blur tonal peaks toward noise
  scan: 0.4,          // drift the capture point so it morphs instead of standing still
});
```

| Option | Default | |
|---|---|---|
| `positionNorm` | `0.5` | Capture point through the source, 0..1. |
| `smear` | `0` | Magnitude box-blur. Values <= 0.01 skip it. |
| `scan` | `0` | How far the capture point drifts over the render. |
| `windowSec` | `0.25` | |
| `seed` | `DEFAULT_SEED` | |
| `onProgress` | -- | Called every 32 hops, then once with `1`. |

### `granular(input, targetSeconds, options?)`

Granular cloud. Dense Hann-windowed grains scattered from a scrub position that advances through the source. Keeps the source's *timbre* intact while dissolving its rhythm, so it sits between PaulStretch's wash and the original material.

Peak-normalised to 0.92.

```js
import { granular } from '@arraypress/paulstretch';

const cloud = await granular(source, 300, {
  grainSeconds: 0.08,
  density: 12,
  panSpread: 0.6,
  pitchSpread: 5,
});
```

| Option | Default | |
|---|---|---|
| `grainSeconds` | `0.15` | Short = pointillist, long = smeared. |
| `density` | `8` | Grains overlapping at any instant. |
| `positionJitter` | `0.05` | Random source-position offset per grain. |
| `timeJitter` | `0.75` | Random onset offset, 0..1 of half the grain spacing. |
| `pitchSpread` | `0` | Random per-grain pitch, ± semitones. |
| `basePitch` | `0` | Pitch applied to every grain, in semitones. |
| `panSpread` | `0` | Random per-grain stereo position. |
| `seed` | `DEFAULT_SEED` | |
| `onProgress` | -- | Called every 256 grains, then once with `1`. |

**Keep `timeJitter` above 0.** At zero, grains fire on a perfectly rigid grid, and when the scrub lingers -- a long target from a short source -- near-identical content re-triggered at exactly `sampleRate / spacing` Hz reads as a buzzy machine-gun. Jitter breaks the grid into an organic rain.

### `phaseVocoder(input, ratio, options?)`

Classic phase-vocoder stretch. Phases are *propagated* between windows from a per-bin instantaneous-frequency estimate rather than randomised, so the source's structure survives: pitch, transients and movement stay recognisable. The engine to reach for when you want material stretched but still *itself*.

**Not normalised** -- the 4x-Hann overlap-add leaves output around 1.5x the source peak. Call `normalizeToPeak` yourself. Nothing is random here, so there is no seed.

```js
import { phaseVocoder, normalizeToPeak } from '@arraypress/paulstretch';

const stretched = await phaseVocoder(source, 8, { pitchSemitones: -5 });
normalizeToPeak(stretched);
```

| Option | Default | |
|---|---|---|
| `windowSec` | `0.25` | |
| `pitchSemitones` | `0` | By spectral bin remapping. |
| `wrapInput` | `false` | |
| `outputFrames` | natural length | Deliver exactly N frames, zero-padding or stopping short. |
| `onProgress` | -- | Called every 16 windows, then once with `1`. |

## Layered drones

Layering thickens a wash the way detuned oscillators thicken a pad: the same source is stretched to different multiples of the target duration and the results are summed.

```js
import { pickLayers, layerSeed, layerPitch, stretch, DEFAULT_SEED } from '@arraypress/paulstretch';

const layers = pickLayers('shimmer');  // null for 'off'

const rendered = [];
for (let i = 0; i < layers.length; i++) {
  rendered.push(await stretch(source, ratio * layers[i].scale, {
    seed: layerSeed(DEFAULT_SEED, i),
    pitchSemitones: layerPitch(0, layers[i]),
  }));
}
// ...then sum at each layer's `gain` and normalise.
```

| Mode | |
|---|---|
| `off` | Single pass. Returns `null`. |
| `subtle` | 3 layers at 0.7x / 1x / 1.4x -- gentle warmth. |
| `standard` | 3 layers at 0.5x / 1x / 2x -- the default thick wash. |
| `lush` | 5 layers, 0.25x to 4x -- deepest static-drone aesthetic. |
| `shimmer` | 3 layers with an octave-up voice -- the Eno-style drone. |
| `shimmerDeep` | Shimmer plus a sub layer an octave down. |

Two things worth knowing:

**Order matters.** Each layer's seed is derived from its *index*, so reordering a recipe changes the audio even though the same layers are present.

**Use `layerSeed`.** Without it every layer draws the identical random sequence, so any two layers whose ratios collide at a cap render bit-identical buffers and sum coherently -- the stack gets louder instead of thicker.

## Progress and yielding

Engines are `async` and take an optional `onProgress`. If the hook returns a promise, the engine awaits it -- which is how a browser caller yields to the event loop mid-render:

```js
await stretch(source, 100, {
  onProgress: (frac) => new Promise((resolve) => {
    setProgress(frac);
    setTimeout(resolve, 0);   // let the browser paint
  }),
});
```

Omit `onProgress` and nothing is called and nothing is awaited -- a Node caller pays no overhead and never yields.

## Buffer helpers

```js
import { peak, normalizeToPeak, frameCount, duration } from '@arraypress/paulstretch';

peak(buffer)                    // largest magnitude across both channels
normalizeToPeak(buffer)         // scale in place so the louder peaks at 0.92
normalizeToPeak(buffer, 1.0)    // ...or at full scale
frameCount(buffer)              // length in frames
duration(buffer)                // length in seconds
```

`normalizeToPeak` applies one shared gain to both channels rather than normalising each independently -- doing it per channel would collapse the stereo image of anything panned off centre.

## Primitives

Exposed because they are useful on their own and because reproducing a render outside this package needs them.

```js
import { fft, FastRNG, blockSeed, mixSeed, nextPow2, DEFAULT_SEED } from '@arraypress/paulstretch';

fft(real, imag, false);   // in-place radix-2. Unscaled forward, 1/N inverse.

const rng = new FastRNG(blockSeed(DEFAULT_SEED, windowIndex));
rng.unit();               // [0, 1)
```

`blockSeed` is what makes range-independent rendering possible: any worker computing window `b` derives the same seed, so windows shared across a chunk boundary come out identical. The double `mixSeed` is deliberate -- seeding adjacent windows with a linear function of the index leaves them correlated, which is audible as amplitude flutter in the overlap-add.

## Tests

```bash
npm test
```

248 tests. The parity suites compare against RMS fingerprints captured from real Swift renders -- 6 cases for `stretch`, 3 each for `freeze`, `granular` and `phaseVocoder`, plus the full layer recipe table. If a fingerprint moves, the port has diverged from the reference and the audio has changed.

## License

MIT
