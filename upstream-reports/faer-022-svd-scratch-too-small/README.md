# faer-022: `svd` panics inside the workspace that `svd_scratch` sized

**What happens:** `linalg::svd::svd` panics with "buffer is not large enough" although its memory comes from `svd_scratch` called with exactly the same arguments.
**Which input:** a single precision (`f32` or `c32`) SVD that uses divide and conquer and does not request `U`. The program below uses an 8 x 8 rank one `f32` matrix, `recursion_threshold = 4`, and `V` only.
**Which version:** `faer` 0.24.4.
**Added to this list:** 2026-10-29.

## Reproduce

```sh
cargo run
```

The matrix has a single entry `A[0,0] = 1`.
The program sets `SvdParams::recursion_threshold = 4`, so the 8 x 8 problem is split once.
It asks `svd_scratch` for the workspace of a call with no `U` and a thin `V`, and then makes that call.
The call must succeed and return `s[0] = 1`.

| | workspace from `svd_scratch` | result |
|---|---|---|
| expected | enough for the call | `Ok`, `s[0] = 1` |
| `faer` 0.24.4 | 3968 bytes | **panic at `src/linalg/svd/bidiag_svd.rs:555`** |

The panic message is `buffer is not large enough to allocate an array of type usize of the requested length: remaining buffer length (after adjusting for alignment): 0, requested array length: 2 (16 bytes)`.
The same happens with `c32` and `A[0,0] = i`.

## Cause

The panic is the allocation of `col_perm` in `compute_svd_of_m` (`src/linalg/svd/bidiag_svd.rs:555`), during a merge step.
The merge allocates several temporaries before it: `col0_perm` and `diag_perm` (lines 504 and 505), and `shifts`, `mus`, `singular_vals` and `zhat` (lines 521 to 527).
The query for this workspace is `divide_and_conquer_scratch` (lines 983 to 1004).
It reserves permutation arrays, Jacobi coefficients and four matrices, but it does not count those column temporaries of the merge.
It also does not reserve room for the recursive calls.
`divide_and_conquer` hands each half of the problem half of the remaining bytes (`stack.make_uninit::<u8>(stack_bytes / 2)`, line 1125), without aligning the split.
So whether the buffer is big enough depends on the element size and on how the bytes divide.

## How the fork fixed it

The fork rewrites the query to follow the live allocations of each phase: deflation, the secular merge temporaries, and the output update.
It reserves two copies of the largest child's unaligned requirement for the recursive halves.
The arithmetic of the SVD is unchanged.
Run against the fork, `svd_scratch` asks for 4288 bytes here and the call returns `Ok` with `s[0] = 1`.

Fork regression test: `linalg::svd::tests::test_v_only_rank_one_exact_scratch` (`f32`, `c32`, `f64`, `c64`), together with `test_divide_and_conquer_exact_scratch` in `linalg::svd::bidiag_svd::tests`.

## Affected versions

Confirmed by running this program: `faer` 0.24.4.
Not fixed upstream: the same code is in upstream `main` at `0ecd846` (2026-10-22).

## References

* Fork fix: tensor4all/faer#80 (commit `36b787c`).
* Where it was found: tensor4all/faer#73.
