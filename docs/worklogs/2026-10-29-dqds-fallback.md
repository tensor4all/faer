# dqds falls back to the rotation path when it does not finish (tensor4all/faer#110)

## Decisions
- **The symptom.** Since #52, the LAPACK SVD suite fails on DCHKBD type 16, seed `0x4c415041434b`, n = 127, 128
  and 129, in `dchkbd_15_to_19` and `bidiagonal_family`. All four dtypes fail. In every failing case the call is
  values-only and returns `NoConvergence`.
  - Bisected: the 0.25.2 release merge `0a4713a` passes; #52's merge `5e45c57` fails.
- **The cause.** Type 16 has entries `eps^U(-2, 2)`, which span 1e-32 to 1e31 in f64.
  - `dqds::singular_values` scales the largest entry to one and squares the entries. A 13-entry block then has
    squared factors near 1e-118.
  - The block's first sweep has a zero shift. Its running pivot `delta` underflows to zero, which the sweep
    rejects. The retries are all at zero shift, so they fail the same way, and the block returns `NoConvergence`.
- **Comparison with LAPACK.** `DBDSQR` with `ROTATE` false calls `DLASQ1`. When that reports that dqds did not
  finish (`INFO = 2`), `DBDSQR` continues with its implicit zero-shift QR on the same bidiagonal.
  - `DLASQ1` also scales the largest entry to `sqrt(eps / safmin)` before squaring, which leaves the most room
    above underflow.
  - That scaling was tried here and does not rescue type 16: the block still fails after one sweep, with factors
    near 1e-120 against a top of about 1e292. The singular values themselves span more than the range of their
    squares. The scaling is therefore not part of this change.
- **The fix.** `compute_bidiag_real_svd` takes the dqds result only when dqds succeeds. Otherwise it continues on
  the rotation path below the values-only branch, the path every call took before #52: the QR iteration below
  the recursion threshold, divide and conquer at or above it.
  - `dqds::singular_values` writes `d` only at the end of a successful run, and it never writes `e`. The fallback
    therefore starts from the caller's bidiagonal. The contract in the function's documentation now says so.

## Verification conclusions and constraints
- **New regular test:** `dchkbd_type_16_values_only_finishes`, in the per-dtype LAPACK test module, not ignored.
  It runs n = 128 at recursion threshold 8 (divide and conquer) and n = 127 at threshold 128 (direct QR), in all
  four `U`/`V` modes. With the fallback removed it fails
  with the original `NoConvergence`.
- **Not added to the regular run:** the whole of `dchkbd_15_to_19`. That suite is opt-in by design (#31); the
  extracted case is the regression.
- **Opt-in suites:** `lapack_svd_*` (`--include-ignored`), including `dchkbd_15_to_19` and `bidiagonal_family`,
  pass for every dtype.
- **No timing claim.** The common path is unchanged, and only inputs that dqds cannot finish take the second
  path.
