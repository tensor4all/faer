# Upstream reports

This directory lists defects that we found while working on a port of `faer` and that also occur in the published upstream crates.
Each entry was reproduced by running its program against the upstream release named in the table.
Defects that exist only in this fork are not listed here.

Each subdirectory is a standalone Cargo project.
It depends only on the upstream crate from crates.io, pinned to the version in the table.
Run it with `cargo run` inside the subdirectory.
The program prints the expected and the actual value, and exits with status 1 when the defect shows.

| # | Title | Upstream crate, version confirmed | Kind of symptom | Fix in this fork | Reported upstream |
|---|---|---|---|---|---|
| [faer-001](faer-001-full-piv-lu-lane-tie/) | Full pivoting LU picks a different pivot depending on the SIMD width | `faer` 0.24.4 | result depends on the CPU | tensor4all/faer#105 (`efaa5de`) | not yet |
| [faer-002](faer-002-partial-svd-residual/) | Partial SVD reports a pair as converged when it is not | `faer` 0.24.4 | wrong value | tensor4all/faer#108 (`1ac38c6`) | not yet |
| [faer-003](faer-003-evd-clamp-grows-with-n/) | The eigenvector residual of the nonsymmetric EVD grows with n | `faer` 0.24.4 | inaccurate value | tensor4all/faer#121 (`0c806f1`) | not yet |
| [gemm-x86-001](gemm-x86-001-diag-negative-stride/) | A product with a diagonal gives a wrong value when the left matrix is stored backwards | `private-gemm-x86` 0.1.22 | wrong value | tensor4all/t4a-private-gemm-x86#5 (`84ca6db`) | not yet |

Each README states, in its first three lines, what goes wrong, for which input and in which version.
It then gives the command, a table of expected and actual output, the cause with file and line in the upstream source, and how this fork fixed it.

Upstream sources:
[faer](https://codeberg.org/sarah-quinones/faer),
[private-gemm-x86](https://github.com/sarah-quinones/gemm-x64-v2).
