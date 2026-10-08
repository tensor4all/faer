# AGENTS.md

Guidance for agents working in this repository.

This is a temporary tensor4all fork of
[faer](https://codeberg.org/sarah-quinones/faer). Read the shared tensor4all
agent rules before acting:
`https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/index.md`
(offline fallback: `../tensor4all-agent-rules/rules/index.md`). Load only the
rule files the task needs.

## Repository-specific rules

* This fork is an AI-driven experiment bench. Experiments are allowed to be
  disruptive. API compatibility with upstream faer is not promised.
* `PATCHSET.md` is the experiment ledger. Record what each experiment does, its
  status, and the tensor4all issue it serves. Keep it current in the same change
  that adds or drops an experiment.
* Keep the rename mechanical so upstream rebases stay cheap:
  * package name `t4a-faer` / `t4a-faer-traits`;
  * `[lib] name = "faer"` / `"faer_traits"`, and dependency keys unchanged, so
    downstream `use faer::...` and `use faer_traits::...` are untouched;
  * `faer-ffi` stays out of the workspace (unused, cbindgen build).
* Do not carry unrelated formatting churn or dependency bumps. Every extra diff
  is a rebase cost against upstream.
* Upstream remote is Codeberg (`git remote add upstream
  https://codeberg.org/sarah-quinones/faer.git`). The GitHub parent
  `sarah-quinones/faer-rs` is a mirror; do not treat it as the source of truth.
* Every experiment carries one focused test. Run it locally; the fork CI gate is
  a build, not the full upstream suite (faer's test dependencies are heavy).
* When an experiment uncovers a likely upstream bug, report it to the fork
  maintainer and get explicit permission before preparing or opening an upstream
  issue or PR.
