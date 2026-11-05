# Complex modulus at the ends of the range (#274)

## Decisions
- `faer-traits`' `abs_impl`, the modulus behind `abs` of `c32`/`c64` (and of
  `ComplexImpl`), the real `math_utils::hypot` and the complex `sqrt`, scaled
  its components by `sqrt_min_positive` above `sqrt_max_positive` and by
  `sqrt_max_positive` below one, then squared them. Below
  `sqrt_min_positive` the squares of a subnormal fall below `min_positive` and
  lose bits or flush to zero (`abs(0 + i 2^-140)` was 0 for `c32`). Above
  about `2^126.5` (`c32`) the scaled squares overflow (`abs(f32::MAX)` was
  `inf`).
- The fix adds a fourth range and widens the two outer scale factors by
  `eps`: `small * eps` above `big`, `big / eps` below `small`, otherwise the
  previous `big`, `1` or `small`. Every factor is a power of two, so in
  `[small, big]` the arithmetic, and the result, are bit for bit the previous
  ones; outside it they differ only where the previous squares left the
  normal range.
- Rejected: keep the old three ranges and recompute in a `#[cold]` function
  when the result is below `2 * min_positive` or not finite. It is correct but
  the call in the loop stops the vectorization of a loop over `abs`: 2.7x
  slower for `c32` (0.41 to 1.10 ns per element) and 1.4x for `c64` on
  Apple M5 Max. A `dlapy2`-style fallback `w sqrt(1 + (z / w)^2)` was also
  dropped: it rounds three times and missed 500,045 of 4,119,171 exact
  Pythagorean `c32` inputs with one subnormal component, against 135,183 for
  the old code.

## Verification conclusions and constraints
- Throughput on Apple M5 Max, 4096 random inputs with `|re|, |im| < 1` and
  `< 100`, median of 21 runs: `c32` 0.41 ns before and after, `c64` 0.82 ns
  before, 0.81 after (std `hypot`: 0.70 / 0.94). Scratch harness, not a
  faer-benchmark campaign; no other host was measured.
- 2,000,000 random finite pairs per type over the whole exponent range: the
  two middle ranges are bit-identical to the old code; outside them every
  result is within one ulp of the correctly rounded modulus (old code: up to
  `inf` at the top, 10 ulp for `c32` and 3 ulp for `c64` at the bottom).
- Not correctly rounded anywhere: in the normal range `sqrt(re^2 + im^2)`
  misses 127,283 of 8,792,209 exact `c32` Pythagorean inputs by one ulp,
  before and after. Bit-equality tests use inputs it gets right.
- Upstream faer carries the same code (`04ef6362`, `06d0265f`, `1498fcc2`);
  upstream was not measured and nothing was reported there.
- Left as they are: the complex `recip` still returns zero for a value whose
  squared magnitude overflows (`recip(f32::MAX + 0i)` is 0, the PATCHSET limit
  of the Schur eigenvector row), and the complex `sqrt` overflows in
  `re + |z|` for `re` near `max_positive`.
