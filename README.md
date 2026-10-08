> **tensor4all による AI 主導の高速実験フォークです。**
>
> [faer](https://codeberg.org/sarah-quinones/faer) の tensor4all フォークです
> （GitHub の `sarah-ek/faer-rs` はミラーです）。ここから公開する `t4a-faer*` crate は
> `tenferro-rs` / `tlinalg-rs` / `tprims-rs` 向けの実験ビルドです。
>
> ### スタンス
>
> - **AI による高速な実験場。** このリポジトリは、コーディングエージェントが高速に
>   反復するための実験場です。コミット・テスト・レビューは素早い試行に最適化されて
>   おり、上流のレビュー基準に合わせることは目的にしていません。
> - **破壊的な実験をする場所。** tensor4all / tenferro のワークロードに最適化した機能
>   — strided・batched のエントリ、型付きの未初期化上書き先、呼び出し側が持つ
>   ワークスペース、明示的なスレッドポリシーなど — を、遠慮なく試して壊します。
>   API の変更と破棄は前提で、互換性は保証しません。
> - **実験の記録。** 何を試し、何を残し、何を捨てたかは
>   [`PATCHSET.md`](./PATCHSET.md) に残します。
>
> ライセンスは上流の著作権表示と出典を保ったまま `MIT` です（[`NOTICE`](./NOTICE)
> 参照）。固定した git revision か、公開済みの `t4a-faer*` 経由でのみ使ってください。

<p align="center">
  <img src="https://faer.veganb.tw/faer-logo-color.png" alt="faer logo"/ width="25%">
</p>

# faer

[![documentation](https://docs.rs/faer/badge.svg)](https://docs.rs/faer)
[![crate](https://img.shields.io/crates/v/faer.svg)](https://crates.io/crates/faer)

`faer` is a rust crate that implements low level linear algebra routines and a high level wrapper for ease of use, in pure rust.
the aim is to provide a fully featured library for linear algebra with focus on portability, correctness, and performance.

see the [official website](https://faer.veganb.tw) and the [docs.rs](https://docs.rs/faer/latest/faer) documentation for code examples and usage instructions.

questions about using the library, contributing, and future directions can be discussed in the [zulip server](https://faer.zulipchat.com).

# contributing

if you'd like to contribute to `faer`, check out the list of "good first issue"
issues. these are all (or should be) issues that are suitable for getting
started, and they generally include a detailed set of instructions for what to
do. please ask questions on the zulip server or the issue itself if anything
is unclear!

# minimum supported rust version

the current msrv is rust 1.84.0.

# benchmarks

see [the benchmark page](https://faer.veganb.tw/benchmarks/) on the main website.

