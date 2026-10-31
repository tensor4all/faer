"""Keep the migration guides identical to the examples and ownership doctest."""

import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
for page, name in [("README", "overview"), ("cpp", "cpp"), ("fortran", "fortran"), ("julia", "julia")]:
    guide = root / "docs" / "migration" / f"{page}.md"
    example = root / "faer" / "examples" / f"migration_{name}.rs"
    blocks = re.findall(r"^```rust\n(.*?)^```", guide.read_text(), re.M | re.S)
    assert len(blocks) == 1, f"{guide}: expected one complete Rust example"
    assert blocks[0].strip() == example.read_text().strip(), f"{guide}: differs from {example}"

index = (root / "docs" / "migration" / "README.md").read_text()
lib_docs = "\n".join(
    line.removeprefix("/// ").removeprefix("///")
    for line in (root / "faer" / "src" / "lib.rs").read_text().splitlines()
)
pattern = r"^```compile_fail,E0505\n(.*?)^```"
ownership = re.findall(pattern, index, re.M | re.S)
assert len(ownership) == 1, "migration README: expected one ownership compile_fail example"
assert ownership == re.findall(pattern, lib_docs, re.M | re.S), (
    "migration README: ownership example differs from the compiled Par doctest"
)
print("migration-examples-ok: 4 examples and the ownership doctest match their guides")
