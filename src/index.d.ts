/**
 * @arraypress/paulstretch — TypeScript definitions.
 */

/**
 * The neutral stereo buffer every engine consumes and produces.
 *
 * `r` may be the SAME array as `l` for mono sources; engines handle that
 * without copying and never write through it. Both channels are expected to be
 * the same length — engines size their loops from `l`, so a short `r` reads
 * past its end and yields NaN.
 */
export interface StereoBuffer {
  l: Float32Array;
  r: Float32Array;
  sampleRate: number;
}

/** One pass of a layered render. `pitchSemitones` stacks on the global shift. */
export interface StretchLayer {
  scale: number;
  gain: number;
  pitchSemitones?: number;
}

/** Named layered-drone recipes, mirroring `LayerPreset` in the Swift reference. */
export type LayerMode = 'off' | 'subtle' | 'standard' | 'lush' | 'shimmer' | 'shimmerDeep';

/**
 * Progress hook. Called with a 0..1 fraction at each engine's own interval, and
 * once with `1` on completion. Return a promise to make the engine await it —
 * that is how a browser caller yields to the event loop mid-render. Omit it
 * entirely and nothing is called or awaited.
 */
export type ProgressHook = (frac: number) => void | Promise<void>;

// ── Buffers ──────────────────────────────────────────────────────────────────

/** Allocate a silent stereo buffer with two independent channels. */
export function createBuffer(frameCount: number, sampleRate: number): StereoBuffer;

/** Build a buffer from existing channels. Omitting `r` produces mono (shared array). */
export function fromChannels(l: Float32Array, r?: Float32Array, sampleRate?: number): StereoBuffer;

/** Length in frames. */
export function frameCount(buffer: StereoBuffer): number;

/** Length in seconds. 0 when the sample rate is not positive. */
export function duration(buffer: StereoBuffer): number;

/** Largest absolute sample across both channels. 0 for silence. */
export function peak(buffer: StereoBuffer): number;

/** Scale both channels in place by ONE gain so the louder peaks at `target` (default 0.92). */
export function normalizeToPeak(buffer: StereoBuffer, target?: number): StereoBuffer;

// ── Primitives ───────────────────────────────────────────────────────────────

/** In-place radix-2 FFT. Unscaled forward, 1/N inverse. Length must be a power of two. */
export function fft(real: Float32Array, imag: Float32Array, inverse: boolean): void;

/** splitmix64 finaliser — decorrelates linear inputs into full-entropy values. */
export function mixSeed(x: bigint): bigint;

/** Deterministic, well-mixed seed for window/grain/hop `b` of a render seeded with `base`. */
export function blockSeed(base: bigint, b: number): bigint;

/** xorshift64 mapped to [0, 1). A zero seed is remapped to the golden-ratio constant. */
export class FastRNG {
  constructor(seed: bigint);
  unit(): number;
}

/** Smallest power of two >= x. Returns 1 for x <= 1. */
export function nextPow2(x: number): number;

// ── Engines ──────────────────────────────────────────────────────────────────

/** The seed every render uses unless one is supplied. Matches `PaulStretcher.defaultSeed`. */
export const DEFAULT_SEED: bigint;

export interface StretchOptions {
  /** STFT window in seconds. Bigger = washier and more diffuse. Default 0.25. */
  windowSec?: number;
  /** Phase randomisation, 0..1. 1 is the classic wash, 0 leaves phase untouched. Default 1. */
  phaseRandomness?: number;
  /** Pitch shift in semitones, applied by resampling. Output length is preserved. Default 0. */
  pitchSemitones?: number;
  /** How strongly rising-energy moments ease off the scramble, 0..1. Default 0. */
  onsetSensitivity?: number;
  /** How far the analysis read head wanders from linear, 0..1. Default 0. */
  spectralDrift?: number;
  /** How much of the previous window's phase carries over, 0..1. Only bites at
   *  full `phaseRandomness`. Default 0. */
  phaseContinuity?: number;
  /** Read the source circularly instead of zero-padding, for seamless loops. Default false. */
  wrapInput?: boolean;
  /** Render seed. Default `DEFAULT_SEED`. */
  seed?: bigint;
  /** Called every 64 windows, then once with 1. */
  onProgress?: ProgressHook;
}

/**
 * PaulStretch: phase-randomised STFT time stretch. Returns audio ~`ratio` times
 * longer, peak-normalised to 0.92. A `ratio` of 1.001 or below returns an
 * unmodified copy.
 */
export function stretch(
  input: StereoBuffer,
  ratio: number,
  options?: StretchOptions,
): Promise<StereoBuffer>;

export interface FreezeOptions {
  /** Capture point through the source, 0..1. Default 0.5. */
  positionNorm?: number;
  /** Magnitude box-blur, 0..1. Values <= 0.01 skip the blur. Default 0. */
  smear?: number;
  /** How far the capture point drifts over the render, 0..1. Default 0. */
  scan?: number;
  /** STFT window in seconds. Default 0.25. */
  windowSec?: number;
  /** Render seed. Default `DEFAULT_SEED`. */
  seed?: bigint;
  /** Called every 32 hops, then once with 1. */
  onProgress?: ProgressHook;
}

/**
 * Spectral freeze: captures one magnitude spectrum and resynthesises it forever
 * with fresh random phase per hop. Peak-normalised to 0.92. Sources under 32
 * frames yield silence; output is never shorter than one window.
 */
export function freeze(
  input: StereoBuffer,
  targetSeconds: number,
  options?: FreezeOptions,
): Promise<StereoBuffer>;

export interface GranularOptions {
  /** Grain length in seconds. Floored at 5 ms, then at 64 frames. Default 0.15. */
  grainSeconds?: number;
  /** Grains overlapping at any instant. Floored at 1. Default 8. */
  density?: number;
  /** Random source-position offset per grain, as a fraction of the source, 0..1. Default 0.05. */
  positionJitter?: number;
  /** Random onset offset per grain, 0..1 of half the grain spacing. Keep above 0 —
   *  a rigid grid re-triggers near-identical content at an exact rate and buzzes. Default 0.75. */
  timeJitter?: number;
  /** Random per-grain pitch, ± semitones. Default 0. */
  pitchSpread?: number;
  /** Pitch applied to every grain, in semitones. Default 0. */
  basePitch?: number;
  /** Random per-grain stereo position, 0..1. Default 0. */
  panSpread?: number;
  /** Render seed. Default `DEFAULT_SEED`. */
  seed?: bigint;
  /** Called every 256 grains, then once with 1. */
  onProgress?: ProgressHook;
}

/**
 * Granular cloud: dense Hann grains scattered from a scrub position that
 * advances through the source. Keeps the source's timbre while dissolving its
 * rhythm. Peak-normalised to 0.92.
 */
export function granular(
  input: StereoBuffer,
  targetSeconds: number,
  options?: GranularOptions,
): Promise<StereoBuffer>;

export interface PhaseVocoderOptions {
  /** STFT window in seconds. Default 0.25. */
  windowSec?: number;
  /** Pitch shift in semitones, by spectral bin remapping. Default 0. */
  pitchSemitones?: number;
  /** Read the source circularly instead of zero-padding, for seamless loops. Default false. */
  wrapInput?: boolean;
  /** Deliver exactly this many frames, zero-padding or stopping short as needed.
   *  Defaults to the engine's natural length. */
  outputFrames?: number;
  /** Called every 16 windows, then once with 1. */
  onProgress?: ProgressHook;
}

/**
 * Phase vocoder: propagates phase between windows instead of randomising it, so
 * the source stays recognisable — pitch, transients and movement survive.
 *
 * NOT normalised. The 4x-Hann overlap-add leaves output around 1.5x the source
 * peak; call `normalizeToPeak` yourself. Nothing is random, so there is no seed.
 */
export function phaseVocoder(
  input: StereoBuffer,
  ratio: number,
  options?: PhaseVocoderOptions,
): Promise<StereoBuffer>;

// ── Layered recipes ──────────────────────────────────────────────────────────

/**
 * The canonical layer recipes. `off` is `null`. Deep-frozen — use `pickLayers`
 * for a mutable copy.
 */
export const LAYER_PRESETS: Readonly<Record<LayerMode, ReadonlyArray<StretchLayer> | null>>;

/**
 * Resolve a mode to a fresh mutable copy of its recipe. `null` for `'off'`; any
 * unknown mode falls back to `'standard'`.
 *
 * Order matters — each layer's seed comes from its INDEX, so reordering a
 * recipe changes the audio even with the same layers present.
 */
export function pickLayers(mode: string): StretchLayer[] | null;

/**
 * Decorrelated seed for layer `index`.
 *
 * Without this every layer draws the identical random sequence, so layers whose
 * ratios collide at a cap render bit-identical buffers and sum coherently — the
 * stack gets louder rather than thicker.
 */
export function layerSeed(baseSeed: bigint, index: number): bigint;

/** A layer's pitch offset stacked on the render's global shift. */
export function layerPitch(basePitchSemitones: number, layer: StretchLayer): number;
