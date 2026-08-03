/**
 * Layered-drone recipes.
 *
 * A layered PaulStretch render runs the whole stretch pipeline several times
 * over the same source — each pass stretched to a different multiple of the
 * target duration, optionally pitch-shifted, then summed. That thickens the
 * wash the way detuned oscillators thicken a pad: the passes drift at
 * different speeds, so their peaks never line up for long.
 *
 * This module is the recipe data plus the two derivations a caller needs to
 * execute one. It deliberately does NOT mix or tile anything — how the layers
 * are summed, trimmed and normalised is the renderer's business, and keeping
 * the recipes pure data means they can be inspected, serialised into a preset
 * and diffed against `LayerPreset.swift` without dragging a render engine in.
 *
 * The values are transcribed from `Sources/PaulStretch/Enums/LayerPreset.swift`
 * in the Swift reference implementation. If that file changes,
 * `tests/layers.test.js` is the thing that should fail.
 */

/**
 * One pass of a layered render.
 *
 * @typedef {object} StretchLayer
 * @property {number} scale Multiple of the target duration this layer
 *   stretches toward before being tiled to fit. Above `1` evolves more slowly
 *   than the base voice, below `1` faster.
 * @property {number} gain Mix gain applied to this layer before the layers are
 *   summed.
 * @property {number} [pitchSemitones] Semitone offset for this layer, ADDED to
 *   the render's global shift — see {@link layerPitch}. Omitted or `0` means
 *   the layer sits at the render's pitch. `+12` is the classic shimmer octave.
 */

/**
 * The built-in layer recipes, keyed by layering mode.
 *
 * ORDER MATTERS. Each layer's phase seed is derived from its INDEX (see
 * {@link layerSeed}), so reordering a recipe changes the rendered audio even
 * though exactly the same layers are present. Treat each array as an ordered
 * score, not a set.
 *
 * The three same-pitch recipes differ only in how many passes stack and how
 * wide their stretch ratios spread:
 *
 * - `off` — `null`. A single unscaled, unity-gain, unshifted pass: the purest
 *   PaulStretch sound, and the fastest.
 * - `subtle` — three passes at 0.7× / 1.0× / 1.4×, centre-heavy and gentle.
 * - `standard` — three passes at 0.5× / 1.0× / 2.0×, the default thick wash.
 * - `lush` — five passes from 0.25× to 4.0×, the densest and slowest-moving
 *   drone, and roughly 5× the render cost of `off`.
 *
 * The two shimmer recipes add a pitched voice. That is only viable because
 * pitch is applied by resampling the source: under the older FFT-bin-remap
 * shift each pitched copy smeared differently window to window and the stack
 * scintillated, which is why layers used to be forced to a single pitch.
 *
 * - `shimmer` — an octave-up voice over the unshifted wash (the Eno drone).
 * - `shimmerDeep` — `shimmer` with the slow layer dropped an octave as well:
 *   wider and darker, with a sub foundation under the rising voice.
 *
 * Frozen, because these are the canonical tables — use {@link pickLayers},
 * which hands back a mutable copy, if you want to tweak a recipe.
 *
 * @type {Readonly<Record<string, ReadonlyArray<StretchLayer>|null>>}
 *
 * @example
 * LAYER_PRESETS.shimmer[2];  // { scale: 1, gain: 0.45, pitchSemitones: 12 }
 * Object.keys(LAYER_PRESETS);
 * // ['off', 'subtle', 'standard', 'lush', 'shimmer', 'shimmerDeep']
 */
export const LAYER_PRESETS = Object.freeze({
  off: null,

  subtle: Object.freeze([
    Object.freeze({ scale: 0.7, gain: 0.45, pitchSemitones: 0 }),
    Object.freeze({ scale: 1.0, gain: 0.75, pitchSemitones: 0 }),
    Object.freeze({ scale: 1.4, gain: 0.45, pitchSemitones: 0 }),
  ]),

  standard: Object.freeze([
    Object.freeze({ scale: 0.5, gain: 0.55, pitchSemitones: 0 }),
    Object.freeze({ scale: 1.0, gain: 0.70, pitchSemitones: 0 }),
    Object.freeze({ scale: 2.0, gain: 0.50, pitchSemitones: 0 }),
  ]),

  lush: Object.freeze([
    Object.freeze({ scale: 0.25, gain: 0.40, pitchSemitones: 0 }),
    Object.freeze({ scale: 0.5, gain: 0.55, pitchSemitones: 0 }),
    Object.freeze({ scale: 1.0, gain: 0.70, pitchSemitones: 0 }),
    Object.freeze({ scale: 2.0, gain: 0.55, pitchSemitones: 0 }),
    Object.freeze({ scale: 4.0, gain: 0.40, pitchSemitones: 0 }),
  ]),

  shimmer: Object.freeze([
    Object.freeze({ scale: 0.5, gain: 0.55, pitchSemitones: 0 }),
    Object.freeze({ scale: 1.0, gain: 0.70, pitchSemitones: 0 }),
    Object.freeze({ scale: 1.0, gain: 0.45, pitchSemitones: 12 }),
  ]),

  shimmerDeep: Object.freeze([
    Object.freeze({ scale: 0.5, gain: 0.50, pitchSemitones: -12 }),
    Object.freeze({ scale: 1.0, gain: 0.70, pitchSemitones: 0 }),
    Object.freeze({ scale: 1.0, gain: 0.45, pitchSemitones: 12 }),
  ]),
});

/**
 * Resolve the layer recipe for a layering mode.
 *
 * Returns `null` for `'off'` so callers can short-circuit to the single-pass
 * render instead of running a one-layer mix. Any unrecognised mode falls back
 * to `'standard'` — a stored preset naming a mode this version doesn't know
 * should still render something musical rather than throw.
 *
 * The returned array is a fresh shallow copy of the preset's layers, mirroring
 * the value semantics of the Swift original: mutate it freely to derive a
 * custom recipe (slow a shimmer voice down by raising its `scale`, say)
 * without corrupting the shared table.
 *
 * ORDER MATTERS — the array is ordered, and layer index feeds the phase seed
 * (see {@link layerSeed}). Reordering the result changes the audio.
 *
 * @param {string} mode - One of `'off'`, `'subtle'`, `'standard'`, `'lush'`,
 *   `'shimmer'`, `'shimmerDeep'`. Anything else resolves to `'standard'`.
 * @returns {StretchLayer[]|null} The recipe in render order, or `null` for
 *   `'off'`.
 *
 * @example
 * pickLayers('off');       // null — render a single unlayered pass
 * pickLayers('standard');  // 3 layers: 0.5× / 1.0× / 2.0×
 *
 * @example
 * // Derive a custom recipe: a shimmer whose octave voice drifts at half speed.
 * const layers = pickLayers('shimmer');
 * layers[2].scale = 2.0;
 */
export function pickLayers(mode) {
  if (mode === 'off') return null;
  const recipe = Object.hasOwn(LAYER_PRESETS, mode) && LAYER_PRESETS[mode]
    ? LAYER_PRESETS[mode]
    : LAYER_PRESETS.standard;
  return recipe.map((layer) => ({ ...layer }));
}

/**
 * The phase seed for layer `index` of a render seeded with `baseSeed`.
 *
 * Decorrelating the layers is not cosmetic. Every layer stretches the SAME
 * source, and the stretch ratios are clamped at `maxStretch` — so two layers
 * whose ratios collide at that cap would draw the identical per-window random
 * sequence, render bit-identical buffers, and sum coherently. The stack gets
 * louder instead of thicker, which is the exact opposite of why you layered
 * it. Striding the seed by the golden-ratio constant gives each layer its own
 * phase stream, so they sum incoherently and thicken.
 *
 * Layer 0 gets `baseSeed` unchanged, so a one-layer recipe renders identically
 * to an unlayered pass at the same seed. Matches `RenderPlan.swift`:
 * `seed &+ UInt64(i) &* 0x9E3779B97F4A7C15`, with the same wrap-around.
 *
 * @param {bigint} baseSeed - The render seed, as a 64-bit `BigInt`. Must be a
 *   `bigint` — mixing in a `Number` throws.
 * @param {number} index - Zero-based layer index (an integer).
 * @returns {bigint} The 64-bit seed for that layer, in `[0, 2^64)`.
 *
 * @example
 * const base = 0x2545f4914f6cdd1dn;  // DEFAULT_SEED
 * layerSeed(base, 0);  // 2685821657736338717n — layer 0 is the base seed
 * layerSeed(base, 1);  // 14086536477059537202n
 * layerSeed(base, 2);  // 7040507222673184071n
 */
export function layerSeed(baseSeed, index) {
  return BigInt.asUintN(64, baseSeed + BigInt(index) * 0x9e3779b97f4a7c15n);
}

/**
 * The absolute pitch, in semitones, at which a layer should be rendered.
 *
 * A layer's offset STACKS on the render's global shift rather than replacing
 * it — that is what keeps a shimmer recipe musical when the whole drone is
 * transposed: pitch the render down 5 semitones and the octave voice follows,
 * staying an octave above the wash instead of drifting into a random interval.
 * Mirrors `p.pitchSemitones + recipe.pitchSemitones` in the Swift plan builder.
 *
 * @param {number} basePitchSemitones - The render's global pitch shift in
 *   semitones.
 * @param {StretchLayer} layer - The layer. A missing `pitchSemitones` counts
 *   as 0.
 * @returns {number} Semitones to render this layer at.
 *
 * @example
 * layerPitch(0, { scale: 1, gain: 0.45, pitchSemitones: 12 });   // 12
 * layerPitch(-5, { scale: 1, gain: 0.45, pitchSemitones: 12 });  // 7
 * layerPitch(-5, { scale: 1, gain: 0.7 });                       // -5
 */
export function layerPitch(basePitchSemitones, layer) {
  return basePitchSemitones + (layer.pitchSemitones ?? 0);
}
