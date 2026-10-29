# faer-029: real Schur QR does not return when the deflation-window sort cannot swap a pair

**What happens:** `Mat::eigen` never returns for some finite `f32` matrices; the thread spins in the real Schur QR.
**Which input:** the near-Jordan matrix `J_n + 1e-6 G` (Jordan block with eigenvalue 1, `G` standard normal from `StdRng` seed 112), divided by its largest entry and scaled by `2^-100`, at n = 128 and n = 257.
**Which version:** `faer` 0.24.4.
**Added to this list:** 2026-10-29.

## Reproduce

```sh
cargo run --release
```

The program runs `Mat::eigen` on that matrix for n = 128 and n = 257 in a separate thread and waits 60 s for each.

| n | expected | `faer` 0.24.4 |
|---|---|---|
| 128 | returns | **no return after 60 s** |
| 257 | returns | **no return after 60 s** |

The same matrices at scales `2^0` to `2^-90` return.
A backtrace taken while it spins ends in `schur_swap_22` called from `aggressive_early_deflation`.

## Cause

`aggressive_early_deflation` sorts the eigenvalues of the deflation window with a bubble sort (`src/linalg/evd/schur/real_schur.rs:921-969`).
It is a port of LAPACK `dlaqr3`, with two differences:

* `dlaqr3` sorts the undeflated part `T(INFQR+1:NS)`. The port sorts from `ns` to `jw`, the deflated part: it took `dlaqr3`'s initial `I = NS + 1`, which only sets the first pass's end `KEND = I - 1`, as the start of the sort.
* `dlaqr3` ends each pass at `KEND`, one before where the previous pass stopped, so the window shrinks by at least one per pass and the sort ends whatever the swaps do. The port sets the next window from the position of the last swap.

`schur_swap_22` rejects an ill-conditioned swap of two 2 x 2 blocks and leaves the matrix unchanged, but `schur_swap` reports success either way.
The pass then records the same swap position as the next window, the pair is still out of order, and the next pass makes the same attempt, forever.

## How the fork fixed it

The sort follows `dlaqr3`: it sorts the undeflated part and bounds each pass by `KEND`.
It swaps with the checked swap, which uses the `dlaexc` ports for the 1 x 2 and 2 x 1 cases, and a rejected swap leaves the pair in place, as `dlaqr3` does after `DTREXC` fails.
With that change the fork returns for both sizes within a few seconds.
At this scale the input lies below LAPACK's negligibility level `smlnum = safmin * n / eps`, so the Schur form reproduces it only to within `smlnum` in absolute terms, as LAPACK's `?HSEQR` does for unscaled input; the same matrix scaled to `2^-80` gives the usual small relative residual.

Fork regression test, in `faer/src/linalg/evd/schur/real_schur.rs`: `tests::aed_sort_terminates_when_a_swap_fails`.

## Affected versions

Confirmed by running this program: `faer` 0.24.4.
Not fixed upstream: the same code is in upstream `main` at `0ecd846` (2026-10-22).

## References

* Fork fix: tensor4all/faer#144 (PR to follow).
* Where it was found: tensor4all/faer#112, the LAPACK comparison sweep.
