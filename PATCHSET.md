# PATCHSET

このフォークで試している実験の台帳。上流 `faer` との差分をここに残す。

## 状態の凡例

- **未着手** — 着手前。関連 issue があるものは issue 列に、無いものは「—」で示す。
- **実験中** — ブランチ上で実装中。壊してよい。
- **採用** — tensor4all 側で使い続ける。上流に送るかは別途判断。
- **破棄** — 試したが捨てた。理由を残す。

## フォーク基盤（上流に送る対象ではない）

| 変更 | 内容 |
| --- | --- |
| crate 名 | `faer` → `t4a-faer`、`faer-traits` → `t4a-faer-traits`。`[lib] name` は `faer` / `faer_traits` のままなので、依存側の `use faer::...` は不変。 |
| workspace | `faer-ffi` を exclude（C API と cbindgen build は tensor4all で未使用）。 |
| 公開 | crates.io へ `t4a-faer-traits` → `t4a-faer` の順で公開。 |
| LICENSE | 各 crate ディレクトリに `LICENSE` を複製して同梱（`license-file` は `license` と併用すると Cargo が警告するため）。 |

## 実験

| 実験 | 内容 | 状態 | 関連 | 備考 |
| --- | --- | --- | --- | --- |
| unpivoted QR の列スキップ | Householder QR が `16·(m−k)·ε` 未満の列をゼロ扱いで落とす。tall/wide の `svd` もこの経路を通る。LAPACK には無いスキップ。 | 未着手 | tlinalg-rs#28, tensor4all-rs#836 | 上流のバグ候補。上流への issue/PR はフォークのメンテナ確認後。 |
| 型付き未初期化上書き先 | `MatUninitMut<T>` のような、初期化前の出力を読まない overwrite 専用の宛先を `matmul_with_conj` に渡せるようにする。 | 未着手 | strided-rs#198 | strided-rs 側に Faer 専用の `Unsupported` 分岐がある。 |
| SVD / eigh の native 直接出力 | nbatch=1 で U・固有ベクトルを中間 `Mat` に取ってからコピーしているのをやめ、呼び出し側の出力領域に直接書く。 | 未着手 | tlinalg-rs#25 | 出力の形状・初期化・失敗時の後始末を保つこと。 |
| reflector 列の借用 | コンパクト QR バッファから reflector をスナッチ用 `basis` にコピーしているのをやめ、列分割で借用する。 | 未着手 | tlinalg-rs#17 | 数値・tau 規約・diagonal beta は不変。 |
| スレッドポリシー | 呼び出し側の rayon プール / work-model（lanes × item threads）を `faer::Par` に写す経路。ambient global state に頼らない。 | 未着手 | tenferro-rs#2000, strided-rs#6 | ネスト並列とスレッド予算が主題。 |
| scratch / plan 再利用 | 呼び出しをまたいで workspace・scratch・計画を保持する。 | 未着手 | — | 起票するかは着手時に決める。 |
| strided / batched view | 汎用 stride とバッチを、packing 無しで直接受ける。 | 未着手 | — | — |
| c64 planar | 複素の実部・虚部分離レイアウトを直接受ける。 | 未着手 | — | — |

## 運用

- 1 実験 = 1 ブランチ。上流 release tag に rebase しやすいよう差分は小さく保つ。
- 無関係な format 変更や依存 bump を載せない（rebase コストになる）。
- 各実験は専用テストを持つ。動かした本人がローカルで実行し、PR ゲート（build）とは別に結果を残す。
- 上流に送るつもりのものは、送る前にクリーンアップとレビューを別途行う。
