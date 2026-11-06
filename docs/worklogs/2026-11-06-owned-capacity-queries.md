# Owned capacity queries

## Decisions

- Capacity is read from the owning allocation, including alignment padding.
  Reporting a requested reservation or reconstructing it from a borrowed view
  would lose the actual ownership contract.
- The public metadata getters retain typed shapes without numeric, Clone or
  std bounds. Borrowed views do not gain allocation-capacity methods. Existing
  allocation and resizing semantics are unchanged.
- This is a public metadata extension, using the existing pending-public-API
  lane in PATCHSET. No existing numerical, allocation, parallel or dispatch
  body changes; the getter chain is not used by any existing numerical path.
  The affected numerical benchmark population is empty, so no
  campaign timing or performance improvement is claimed. Source-based constant
  field access is not presented as a measured speed result.

## Verification conclusions and constraints

- Focused public-contract tests cover four scalars with complex values,
  padding, no-op reservation, truncate/regrow, pointer/value preservation,
  empty extents, typed shapes, ownership conversions and a nonnumeric non-Clone
  element. Default and minimal std configurations pass; no_std compilation is
  supported.
- Each getter has an executable usage example. Focused validation and hosted
  results belong to their exact candidate revision and are recorded in the
  proposing PR.
- No kernels or benchmark harnesses are added. The Colmat prototype and its
  final dependency adoption/canonical verification remain downstream work.
- Zero-sized-element allocation is an inherited separate problem tracked by
  #286; these read-only getters do not change allocation behavior or repair it.
