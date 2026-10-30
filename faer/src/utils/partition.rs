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
/// 1)/2`. Boundaries are rounded up to multiples of `align` (the kernel's
/// column step), are nondecreasing, and part `parts - 1` ends at `n`; a part
/// may be empty. An upper triangle is the lower one of the transpose, and a
/// strict triangle shifts the rows by one before calling this.
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
	let (mf, nf) = (m as f64, n.min(m) as f64);
	let total = nf * mf - nf * (nf - 1.0) / 2.0;
	let want = total * i as f64 / parts as f64;
	// c*m - c^2/2 + c/2 = want, the smaller root
	let b = mf + 0.5;
	let c = b - (b * b - 2.0 * want).max(0.0).sqrt();
	((c.round() as usize).next_multiple_of(align.max(1))).min(n)
}

#[cfg(test)]
mod tests {
	use super::lower_area_split;

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
