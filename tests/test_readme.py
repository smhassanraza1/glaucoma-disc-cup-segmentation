import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KEYS = ("disc_dice", "cup_dice", "disc_iou", "cup_iou", "vcdr_mae", "vcdr_corr")


def block(name: str) -> str:
    found = re.search(rf"<!-- {name}:start -->\n(.*?)<!-- {name}:end -->", (ROOT / "README.md").read_text(), re.DOTALL)
    assert found, f"README.md has no {name} markers"
    return found.group(1).strip()


def test_readme_tables_are_exactly_the_evaluation_output():
    assert block("results") == (ROOT / "results" / "table.md").read_text().strip()
    assert block("bias") == (ROOT / "results" / "bias.md").read_text().strip()


def test_table_matches_metrics_json():
    metrics = json.loads((ROOT / "results" / "metrics.json").read_text())
    rows = block("results").splitlines()
    for label, group in metrics["groups"].items():
        row = next(line for line in rows if line.startswith(f"| {label} |"))
        cells = [c.strip() for c in row.strip("|").split("|")]
        assert int(cells[1]) == group["n_images"]
        for cell, key in zip(cells[2:], KEYS, strict=True):
            assert cell == f"{group[key]:.3f}"
    base = next(line for line in rows if line.startswith("| Baseline"))
    assert f"{metrics['baseline']['vcdr_mae']:.3f}" in base


def test_derived_figures_in_the_prose_match_metrics_json():
    group = json.loads((ROOT / "results" / "metrics.json").read_text())["groups"]["All"]
    readme = (ROOT / "README.md").read_text()
    expected = (
        f"our test results are {group['vcdr_mae']:.3f} for the vCDR error and "
        f"{1 - group['disc_iou']:.3f} and {1 - group['cup_iou']:.3f} for the disc and cup overlap"
    )
    assert expected in " ".join(readme.split())
