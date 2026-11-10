# Probe staged for measurement (not applied to the library)

The measurements in `../` were produced by building this crate next to a checkout of
this repository with `../instrumentation.patch` applied:

    ~/work/eigh391/probe           <- this directory
    ~/work/eigh391/src/faer        <- the faer crate inside that checkout

so the dependency `../src/faer` resolves to the instrumented faer crate when the
probe sits beside the checkout. To reproduce, copy this directory next to a checkout
with the patch applied, keeping that layout, and build with `--features count` for
the count-only cells and `--features instr` (the default) for the timing cells.
