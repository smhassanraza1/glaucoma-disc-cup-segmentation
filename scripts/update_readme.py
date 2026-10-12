"""Copy the evaluation tables into the README, between their markers.

    python scripts/update_readme.py

The README never holds hand-typed numbers: the tables come from the evaluation script's own output
(results/table.md and results/bias.md), and tests/test_readme.py fails if the two ever differ.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BLOCKS = {"results": "table.md", "bias": "bias.md"}


def main() -> None:
    readme = (ROOT / "README.md").read_text()
    for name, filename in BLOCKS.items():
        pattern = re.compile(rf"(<!-- {name}:start -->\n).*?(<!-- {name}:end -->)", re.DOTALL)
        if not pattern.search(readme):
            sys.exit(f"README.md has no {name} markers")
        table = (ROOT / "results" / filename).read_text()
        readme = pattern.sub(lambda m, t=table: m.group(1) + "\n" + t + "\n" + m.group(2), readme)
    (ROOT / "README.md").write_text(readme)
    print("README tables updated")


if __name__ == "__main__":
    main()
