# faer-042: partial pivoting LU warns "generic strides" on contiguous row-major input

**What happens:** with the `perf-warn` feature on, the first column-pivoted LU of a contiguous 64 x 64 row-major matrix logs `LU with partial pivoting prefers column-major or row-major matrix. Found matrix with generic strides.` at target `faer_perf`. This report counts warnings only: it does not check the factors numerically. The message is the defect, and it names row-major as one of the two layouts it says it prefers.
**Which input:** any matrix whose `col_stride` is `1`, i.e. any contiguous row-major one, of any shape. The program uses a `64 x 64` `f64` matrix and passes the transpose view of a column-major `Mat`, so `row_stride = 64`, `col_stride = 1`.
**Which version:** `faer` 0.24.4, and upstream `main` at `0ecd846` (2026-10-22), where the predicate is unchanged.
**Added to this list:** 2026-11-09.

## Reproduce

```sh
cargo run                    # contiguous row-major: exits with status 1
cargo run -- column-major    # the same matrix in the other preferred layout: exits 0
```

The program installs a small `log::Log` that counts records at target `faer_perf`, runs one partial-pivoting `lu_in_place`, and prints the layout, the two strides, and the expected and actual warning counts.
`__perf_warn!` is a process-wide "warn once" flag, so each run is one factorization: the row-major run is the default and the `column-major` run is the control, in its own process.

| input | expected warnings | `faer` 0.24.4 |
|---|---|---|
| `64 x 64` contiguous row-major (`row_stride` 64, `col_stride` 1) | 0 | **1** |
| `64 x 64` contiguous column-major (`row_stride` 1, `col_stride` 64) | 0 | 0 |

The column-major run not warning is what hid the defect: the predicate is only wrong for the other one of the two layouts its message recommends.

## Cause

```text
src/linalg/lu/partial_pivoting/factor.rs:245-253   (faer 0.24.4; main 0ecd846 has it at the same lines)
	if (A.col_stride().unsigned_abs() == 1
		|| A.row_stride().unsigned_abs() != 1)
		&& crate::__perf_warn!(LU_WARN)
	{
		log::warn!(
			target : "faer_perf",
			"LU with partial pivoting prefers column-major or row-major matrix. Found matrix with generic strides."
		);
	}
```

The condition is `col_stride == 1 || row_stride != 1`, which is "anything that is not contiguous column-major": on a contiguous row-major matrix the first clause is true (`col_stride == 1`), and on any matrix that is not column-major the second one is. So the warning fires on every row-major input, which is one of the two layouts the message recommends. The predicate that matches the message is `row_stride != 1 && col_stride != 1` -- neither stride walks a line, so the input is neither layout.

The full-pivoting sibling `src/linalg/lu/full_pivoting/factor.rs:477-485` carries the same condition -- the same predicate in another file, so it is noted here instead of getting its own entry. The fork fixes that copy in tensor4all/faer#342 and this one in tensor4all/faer#345.

## How the fork fixed it

One predicate, used by the warning and covered by a unit test:

```text
faer/src/linalg/lu/partial_pivoting/factor.rs:633-640   (tensor4all/faer, #345)
#[cfg(any(feature = "perf-warn", test))]
#[inline]
pub(crate) fn has_generic_strides(
	row_stride: isize,
	col_stride: isize,
) -> bool {
	row_stride.unsigned_abs() != 1 && col_stride.unsigned_abs() != 1
}
```

`faer/src/linalg/lu/partial_pivoting/factor.rs:649-652` calls it from the `perf-warn` block, and
`faer/src/linalg/lu/partial_pivoting/factor.rs:762-773` asserts the four layouts: contiguous row-major and column-major and a transposed view are silent, and a matrix with neither stride equal to one warns.
The fork's fix is a warning-message fix only; the factorisation is untouched.
