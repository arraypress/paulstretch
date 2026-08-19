/**
 * Deterministic PRNG and seed derivation.
 *
 * Every random decision in this package flows through a seeded generator, so
 * the same source, parameters and seed always produce bit-identical output.
 * That reproducibility is not a nicety — it is what lets a render be split
 * across workers or chunks and still agree on the windows they share, and it
 * is what makes verifying this port against the Swift reference possible at
 * all.
 *
 * JavaScript has no 64-bit integers, so the arithmetic is split two ways:
 *   - `mixSeed` / `blockSeed` use BigInt. Exact, and only called once per
 *     window, so the cost is irrelevant.
 *   - `FastRNG` uses 32-bit hi/lo limbs. Called once per FFT bin, so it has to
 *     stay in the JIT's integer fast path.
 *
 * Both are verified bit-for-bit against the Swift original — see
 * `tests/rng.test.js`.
 */

const MASK64 = (1n << 64n) - 1n;

/**
 * splitmix64 finaliser — turns sequential or otherwise linear inputs into
 * full-entropy, well-decorrelated 64-bit values.
 *
 * Seeding adjacent windows with a *linear* function of the window index gives
 * correlated phase sequences between neighbours, which is audible as amplitude
 * flutter in the overlap-add. Mixing through splitmix64 removes it.
 *
 * @param {bigint} x - Any 64-bit value.
 * @returns {bigint} A well-mixed 64-bit value.
 *
 * @example
 * mixSeed(1n);  // 6238072747940578789n
 */
export function mixSeed(x) {
  let z = (x + 0x9e3779b97f4a7c15n) & MASK64;
  z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & MASK64;
  z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & MASK64;
  return (z ^ (z >> 31n)) & MASK64;
}

/**
 * Deterministic, well-mixed seed for window (or grain, or hop) `b` of a render
 * seeded with `base`.
 *
 * Any worker or chunk computing window `b` derives the same seed, so shared
 * boundary windows come out identical everywhere. The double mix is
 * deliberate: a single pass still leaves neighbours correlated.
 *
 * @param {bigint} base - The render seed.
 * @param {number} b - Window index. May be negative.
 * @returns {bigint} The seed for that window.
 *
 * @example
 * blockSeed(DEFAULT_SEED, 0);
 * blockSeed(DEFAULT_SEED, 41);
 */
export function blockSeed(base, b) {
  const bi = BigInt.asUintN(64, BigInt(b) + 1n);
  return mixSeed((BigInt(base) ^ mixSeed(bi)) & MASK64);
}

/**
 * xorshift64 mapped to `[0, 1)`.
 *
 * State is held as two 32-bit halves. All shifts are the 64-bit ones expressed
 * over the pair, and `unit()` reproduces `Double(s >> 11) * 2^-53` without ever
 * exceeding float64's 53-bit exact-integer range.
 *
 * @example
 * const rng = new FastRNG(blockSeed(DEFAULT_SEED, 0));
 * rng.unit();  // 0..1
 */
export class FastRNG {
  /**
   * @param {bigint|number} seed - 64-bit seed. A number is coerced, so
   *   `seed: 7` works — the engines take it straight from caller options, and
   *   failing there surfaces as `Cannot mix BigInt and other types` from deep
   *   inside the RNG rather than anything the caller can act on. Zero is
   *   remapped to the golden-ratio constant, because xorshift64 is stuck at
   *   zero forever otherwise.
   */
  constructor(seed) {
    const seed64 = BigInt(seed);
    const s = seed64 === 0n ? 0x9e3779b97f4a7c15n : BigInt.asUintN(64, seed64);
    this.hi = Number((s >> 32n) & 0xffffffffn) >>> 0;
    this.lo = Number(s & 0xffffffffn) >>> 0;
  }

  /**
   * The next value in `[0, 1)`.
   *
   * @returns {number} Uniform random value.
   */
  unit() {
    let hi = this.hi;
    let lo = this.lo;

    // s ^= s << 13
    let sh = (hi << 13) | (lo >>> 19);
    let sl = lo << 13;
    hi = (hi ^ sh) >>> 0;
    lo = (lo ^ sl) >>> 0;

    // s ^= s >> 7  (logical — the reference shifts an unsigned 64-bit value)
    sh = hi >>> 7;
    sl = (lo >>> 7) | (hi << 25);
    hi = (hi ^ sh) >>> 0;
    lo = (lo ^ sl) >>> 0;

    // s ^= s << 17
    sh = (hi << 17) | (lo >>> 15);
    sl = lo << 17;
    hi = (hi ^ sh) >>> 0;
    lo = (lo ^ sl) >>> 0;

    this.hi = hi;
    this.lo = lo;

    // Double(s >> 11) * 2^-53. `s >> 11` == hi * 2^21 + (lo >>> 11), which is
    // below 2^53 and therefore exact in float64.
    return (hi * 2097152 + (lo >>> 11)) * (1 / 9007199254740992);
  }
}

/**
 * The smallest power of two greater than or equal to `x`.
 *
 * @param {number} x - Any number. Values below 1 return 1.
 * @returns {number} A power of two.
 *
 * @example
 * nextPow2(1000);  // 1024
 * nextPow2(1024);  // 1024
 * nextPow2(0);     // 1
 */
export function nextPow2(x) {
  let n = 1;
  while (n < x) n <<= 1;
  return n;
}
