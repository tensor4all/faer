# A diagonal product reads the depth in either direction (tensor4all/faer#113)

## Decisions
- **What #113 recorded, and what actually failed.** #113 was filed from `docs/lapack-validation.md`, which
  still described a pivoted-LDL metadata failure: a stale subdiagonal on no-update 1×1 pivots. That failure
  had already been fixed by #67. The diagnostics #113 counted (164 / 240 / 172 / 240 per dtype) were all in
  the *unpivoted* LDL:
  - n of 65 or more;
  - layouts 2 and 3 only, the two that reverse rows and columns;
  - every kind, including the positive definite kind 1;
  - reconstruction, solve and inverse ratios of 1e9 to 1e15.

  Because a positive definite input failed, this was a defect and not instability.
- **The cause.**
  - The unpivoted LDL's blocked step updates the trailing block with `L10 D0 L10^H` through `spicy_matmul`,
    passing the diagonal `D0`. On x86 that calls `t4a-private-gemm-x86`.
  - The kernel turns a negative depth stride of its left operand around: it reverses that operand's columns
    and the right operand's rows (0.1.26, `lib.rs:1408`). It does not reverse the diagonal, so depth index
    `t` met `D[depth - 1 - t]`.
  - The kernel's left operand is `A`, or `B` transposed when the destination's rows are further apart than
    its columns, because it swaps the two first.
  - Reduced to the product: `A D B` is wrong whenever that operand's depth stride is negative, and right in
    every other combination of directions, including without `D`.
- **The fix, in faer.** `spicy_matmul` reverses the depth of `A`, `B` and `D` together when the kernel's left
  operand would have a negative depth stride. That reorders the sum and does not change it. The root cause is
  in the kernel crate (also reverse `real_diag` in that branch); its repository has issues disabled, so the fix
  is described in the faer PR. The workaround can go once a fixed kernel is required.

## Verification conclusions and constraints
- `linalg::matmul::internal::reversed_depth_tests::a_diagonal_product_reads_the_depth_in_any_direction`
  covers both operands in both depth directions, into a column-major and a row-major destination, and is
  exact against the plain sum. Without the fix it fails at `B` reversed into a row-major destination, which is
  the swapped-operand case.
- `lapack_indefinite_*` with `--include-ignored` passes all 12 tests: cases and parallel, in four dtypes.
- Only the x86 kernel route takes a diagonal this way. Other targets go through the generic route, which reads
  `D` by index.
