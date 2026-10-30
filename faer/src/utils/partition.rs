//! Static partitions of a parallel region's work (faer#168).
//!
//! A part computed once from the shape, before the region starts, so every
//! participant knows its share without negotiating it: the static-ownership
//! split OpenBLAS's level-3 drivers use. The first is the triangle
//! (`syrk_thread`), shared by the products into a triangle and, in turn, by the
//! factorizations whose trailing update is one (Cholesky, LU, the symmetric
//! reductions).

/// The first column of part `i` of `parts` when the lower part of an `m x n`
/// matrix (row `r`, column `c`, `r >= c`) is cut into column ranges of equal
/// area.
///
/// Column `c` holds `m - c` entries, so the area left of `c` is `c*m - c*(c -
/// 1)/2`. Boundaries are rounded to the nearest column, then up to multiples of
/// `align` (the kernel's column step), are nondecreasing, and part `parts - 1`
/// ends at `n`; a part may be empty. An upper triangle is the lower one of the
/// transpose, and a strict triangle shifts the rows by one before calling this.
#[cfg_attr(
	not(all(target_arch = "x86_64", feature = "std", feature = "threadteam")),
	allow(dead_code)
)]
pub(crate) fn lower_area_split(
	m: usize,
	n: usize,
	i: usize,
	parts: usize,
	align: usize,
) -> usize {
	if i == 0 {
		return 0;
	}
	if i >= parts {
		return n;
	}
	// In integers, because the crate builds without `std` (no `f64::sqrt`). The
	// boundary is the smaller root `c` of `c*m - c*(c - 1)/2 = want`, `want =
	// total * i / parts`, rounded to the nearest column: with `B = 2m + 1`, `c
	// = (B - sqrt(B^2 - 8 want)) / 2`, so the nearest column is the largest
	// `k` with `parts * (B + 1 - 2k)^2 >= q`, where `q = parts * B^2 - 8 *
	// total * i`.
	let (m, w, p) = (m as u128, n.min(m) as u128, parts as u128);
	let total = w * m - w * w.saturating_sub(1) / 2;
	let big_b = 2 * m + 1;
	let q = p * big_b * big_b - 8 * total * i as u128;
	let fits =
		|k: u128| k <= (big_b + 1) / 2 && p * (big_b + 1 - 2 * k).pow(2) >= q;
	let mut k = (big_b + 1 - (q / p).isqrt()) / 2;
	while fits(k + 1) {
		k += 1;
	}
	while !fits(k) {
		k -= 1;
	}
	(k as usize).next_multiple_of(align.max(1)).min(n)
}

#[cfg(test)]
mod tests {
	use super::lower_area_split;

	/// The boundaries are the nearest columns to the exact roots, decided in
	/// integers (no floating point, so every host cuts the same way).
	/// Reference values from the closed form.
	#[test]
	fn lower_boundaries_are_the_rounded_roots() {
		let cuts = |m, n, parts: usize, count: usize| -> Vec<usize> {
			(1..=count)
				.map(|i| lower_area_split(m, n, i, parts, 1))
				.collect()
		};
		assert_eq!(
			cuts(4096, 4096, 16, 8),
			[130, 265, 404, 549, 700, 858, 1024, 1200]
		);
		assert_eq!(cuts(1024, 1024, 64, 8), [8, 16, 24, 33, 41, 49, 58, 66]);
		assert_eq!(cuts(2047, 2048, 7, 6), [152, 317, 500, 707, 953, 1274]);
	}

	/// The parts cover `0..n` in order, start on multiples of `align`, and hold
	/// equal areas up to one aligned column strip.
	#[test]
	fn lower_parts_cover_the_triangle_with_equal_areas() {
		for &(m, n) in &[
			(4096usize, 4096usize),
			(2047, 2048),
			(301, 301),
			(40, 40),
			(9, 10),
			(900, 300),
		] {
			for align in [1usize, 8] {
				for parts in [1usize, 2, 3, 4, 7, 16, 64] {
					let cuts: Vec<usize> = (0..=parts)
						.map(|i| lower_area_split(m, n, i, parts, align))
						.collect();
					assert_eq!((cuts[0], cuts[parts]), (0, n));
					assert!(
						cuts.windows(2).all(|w| w[0] <= w[1]),
						"{m}x{n}/{parts}: {cuts:?}"
					);
					assert!(
						cuts[1..parts]
							.iter()
							.all(|c| c % align == 0 || *c == n)
					);
					let area = |c0: usize, c1: usize| {
						(c0..c1.min(m)).map(|c| m - c).sum::<usize>()
					};
					let total = area(0, n);
					for w in cuts.windows(2) {
						assert!(
							area(w[0], w[1]).abs_diff(total / parts)
								<= align * m,
							"{m}x{n}/{parts}/{align}: {cuts:?}"
						);
					}
				}
			}
		}
	}
}
