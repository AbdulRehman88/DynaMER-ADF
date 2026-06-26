# scripts/26A_collect_manuscript_locked_results.py
# -*- coding: utf-8 -*-
"""
Stage 26A: Collect manuscript-locked figures, tables, and evidence files.

Purpose
-------
Copy all manuscript-facing materials into one clean folder:

    outputs/manuscript_locked_results/

The script DOES NOT move, delete, or modify original outputs.
It only copies files from their original locations into an organized
manuscript-locked results folder.

Run from project root:
    python scripts/26A_collect_manuscript_locked_results.py --overwrite

Optional:
    python scripts/26A_collect_manuscript_locked_results.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


# -----------------------------
# Locked manuscript material map
# -----------------------------

MAIN_PAPER_FIGURES = [
    # Methods
    "Fig_DynaMER_ADF_architecture.pdf",

    # Results
    "Fig01_balanced_accuracy_final_validation.pdf",
    "Fig25_protocol_gap_ba_macro_f1.pdf",
    "Fig25_modality_class_vs_subject_fisher.pdf",
    "Fig05a_modality_delta_heatmap_polished.pdf",
    "Fig05b_architecture_delta_heatmap_polished.pdf",
]

OPTIONAL_MAIN_OR_SUPPLEMENT_FIGURES = [
    "Fig25_subject_dominance_index_by_modality.pdf",
    "Fig25_loso_subject_difficulty.pdf",
]

SUPPLEMENTARY_FIGURE_PATTERNS = [
    "Fig01_*.*",
    "Fig02_cross_session_*.*",
    "Fig02_subject_loso_*.*",
    "Fig05a_modality_delta_heatmap_polished.*",
    "Fig05b_architecture_delta_heatmap_polished.*",
    "Fig06_fold_stability_boxplot_polished.*",
    "Fig07_efficiency_vs_performance_polished.*",
    "Fig25_subject_dominance_index_by_modality.*",
    "Fig25_loso_subject_difficulty.*",
    "Fig25_confusion_*.*",
    "Fig24_protocol_extension_ba_macro_f1.*",
    "Fig24_nested_loso_selected_variant_counts.*",
]

MAIN_AND_SUPPLEMENT_TABLE_PATTERNS = [
    # Existing manuscript-ready tables
    "*.tex",
    "*.csv",
    "*.json",

    # Stage 22 subject-mixed
    "22C_*.csv",
    "22D_*.csv",
    "22D_*.json",
    "22F_*.csv",
    "22F_*.tex",
    "22F_*.json",
    "22G_*.csv",
    "22G_*.tex",
    "22G_*.json",

    # Stage 23 nested LOSO
    "23B_*.csv",
    "23B_*.tex",
    "23B_*.json",

    # Stage 24 protocol package
    "24A_*.csv",
    "24A_*.tex",
    "24A_*.json",
    "24A_*.md",

    # Stage 25 physiological evidence
    "25A_*.csv",
    "25A_*.tex",
    "25A_*.json",
    "25A_*.md",
]

SOURCE_ROOTS_RELATIVE = [
    "figures",
    "tables",
    "outputs/manuscript_ready_assets",
    "outputs/final_locked_results",
    "outputs/protocol_extension/22_seed_iv_subject_mixed_5fold",
    "outputs/protocol_extension/23_nested_loso_dynamer_adf",
    "outputs/protocol_extension/24_protocol_extension_evidence_package",
    "outputs/physiology_evidence/25_physiological_evidence_audit",
]


@dataclass
class CopyRecord:
    category: str
    filename: str
    source: str
    destination: str
    status: str
    sha256: str = ""
    note: str = ""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def iter_existing_roots(project_root: Path, out_root: Path | None = None) -> list[Path]:
    roots = []
    out_resolved = out_root.resolve() if out_root is not None else None

    for rel in SOURCE_ROOTS_RELATIVE:
        p = (project_root / rel).resolve()
        if not p.exists():
            continue

        # Never scan the destination folder itself.
        if out_resolved is not None:
            try:
                p.relative_to(out_resolved)
                continue
            except ValueError:
                pass

        roots.append(p)

    # IMPORTANT:
    # Do NOT add the whole outputs/ folder as a catch-all source,
    # because manuscript_locked_results is also inside outputs/.
    return roots


def find_matches(roots: Iterable[Path], pattern: str) -> list[Path]:
    matches: list[Path] = []
    for root in roots:
        matches.extend([p for p in root.rglob(pattern) if p.is_file()])
    # Prefer PDFs over raster if duplicate base name is requested later.
    return sorted(set(matches), key=lambda p: (p.name.lower(), len(str(p))))


def find_first_by_name(roots: Iterable[Path], filename: str) -> Path | None:
    matches = []
    for root in roots:
        matches.extend([p for p in root.rglob(filename) if p.is_file()])
    if not matches:
        return None

    def priority(p: Path) -> tuple[int, int]:
        s = str(p).replace("\\", "/").lower()
        score = 100
        if "manuscript_ready_assets" in s:
            score -= 30
        if "physiology_evidence" in s:
            score -= 25
        if "protocol_extension" in s:
            score -= 20
        if "/figures/" in s:
            score -= 10
        if p.suffix.lower() == ".pdf":
            score -= 5
        return score, len(s)

    return sorted(matches, key=priority)[0]


def copy_file(
    src: Path,
    dst: Path,
    category: str,
    overwrite: bool,
    dry_run: bool,
    records: list[CopyRecord],
    note: str = "",
) -> None:
    ensure_dir(dst.parent)
    try:
        if src.resolve() == dst.resolve():
            records.append(
                CopyRecord(
                    category=category,
                    filename=src.name,
                    source=str(src),
                    destination=str(dst),
                    status="skipped_same_file",
                    sha256=sha256_file(src),
                    note="Source and destination are the same file.",
                )
            )
            return
    except FileNotFoundError:
        pass

    if dst.exists() and not overwrite:
        records.append(
            CopyRecord(
                category=category,
                filename=src.name,
                source=str(src),
                destination=str(dst),
                status="skipped_exists",
                sha256=sha256_file(dst),
                note="Destination exists. Use --overwrite to replace.",
            )
        )
        return

    if not dry_run:
        shutil.copy2(src, dst)
        digest = sha256_file(dst)
    else:
        digest = sha256_file(src)

    records.append(
        CopyRecord(
            category=category,
            filename=src.name,
            source=str(src),
            destination=str(dst),
            status="copied_dry_run" if dry_run else "copied",
            sha256=digest,
            note=note,
        )
    )


def copy_named_files(
    roots: list[Path],
    filenames: list[str],
    dst_dir: Path,
    category: str,
    overwrite: bool,
    dry_run: bool,
    records: list[CopyRecord],
) -> None:
    for fname in filenames:
        src = find_first_by_name(roots, fname)
        if src is None:
            records.append(
                CopyRecord(
                    category=category,
                    filename=fname,
                    source="",
                    destination=str(dst_dir / fname),
                    status="missing",
                    note="File not found under configured source roots.",
                )
            )
            continue
        copy_file(src, dst_dir / src.name, category, overwrite, dry_run, records)


def copy_pattern_files(
    roots: list[Path],
    patterns: list[str],
    dst_dir: Path,
    category: str,
    overwrite: bool,
    dry_run: bool,
    records: list[CopyRecord],
    allowed_suffixes: set[str] | None = None,
) -> None:
    seen: set[Path] = set()
    for pattern in patterns:
        for src in find_matches(roots, pattern):
            if allowed_suffixes and src.suffix.lower() not in allowed_suffixes:
                continue
            if src in seen:
                continue
            seen.add(src)

            # Preserve a small amount of source context to avoid collisions.
            parent_label = src.parent.name
            dst = dst_dir / parent_label / src.name
            copy_file(src, dst, category, overwrite, dry_run, records, note=f"Matched pattern: {pattern}")


def copy_source_evidence_folders(
    project_root: Path,
    out_root: Path,
    overwrite: bool,
    dry_run: bool,
    records: list[CopyRecord],
) -> None:
    folder_map = {
        "manuscript_ready_assets": project_root / "outputs" / "manuscript_ready_assets",
        "stage22_subject_mixed_5fold": project_root / "outputs" / "protocol_extension" / "22_seed_iv_subject_mixed_5fold",
        "stage23_nested_loso": project_root / "outputs" / "protocol_extension" / "23_nested_loso_dynamer_adf",
        "stage24_protocol_extension": project_root / "outputs" / "protocol_extension" / "24_protocol_extension_evidence_package",
        "stage25_physiology_evidence": project_root / "outputs" / "physiology_evidence" / "25_physiological_evidence_audit",
        "final_locked_results": project_root / "outputs" / "final_locked_results",
    }

    allowed = {".csv", ".json", ".tex", ".md", ".txt", ".pdf", ".png", ".svg"}

    for label, folder in folder_map.items():
        if not folder.exists():
            records.append(
                CopyRecord(
                    category="source_evidence_folder",
                    filename=label,
                    source=str(folder),
                    destination=str(out_root / "source_evidence" / label),
                    status="missing_folder",
                )
            )
            continue

        for src in folder.rglob("*"):
            if not src.is_file() or src.suffix.lower() not in allowed:
                continue
            rel = src.relative_to(folder)
            dst = out_root / "source_evidence" / label / rel
            copy_file(src, dst, "source_evidence_folder", overwrite, dry_run, records)


def write_inventory(records: list[CopyRecord], out_root: Path, dry_run: bool) -> None:
    inventory_dir = out_root / "inventory"
    ensure_dir(inventory_dir)

    csv_path = inventory_dir / "manuscript_locked_results_inventory.csv"
    json_path = inventory_dir / "manuscript_locked_results_inventory.json"
    missing_path = inventory_dir / "missing_or_skipped_files.csv"

    rows = [r.__dict__ for r in records]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["category", "filename", "source", "destination", "status", "sha256", "note"],
        )
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "dry_run": dry_run,
                "total_records": len(records),
                "copied": sum(r.status in {"copied", "copied_dry_run"} for r in records),
                "missing": sum(r.status in {"missing", "missing_folder"} for r in records),
                "skipped_exists": sum(r.status == "skipped_exists" for r in records),
                "records": rows,
            },
            f,
            indent=2,
        )

    bad = [r for r in records if r.status not in {"copied", "copied_dry_run"}]
    with missing_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["category", "filename", "source", "destination", "status", "sha256", "note"],
        )
        writer.writeheader()
        writer.writerows([r.__dict__ for r in bad])


def write_readme(out_root: Path) -> None:
    readme = out_root / "README_MANUSCRIPT_LOCKED_RESULTS.md"
    text = f"""# Manuscript Locked Results Folder

Created: {datetime.now().isoformat(timespec="seconds")}

This folder is a copied, organized snapshot of manuscript-facing results.
Original files remain in their original locations.

## Main folders

- `main_paper/figures/`
  - Figures intended for the main manuscript.
- `supplementary/figures/`
  - Candidate supplementary figures.
- `supplementary/tables/`
  - Candidate supplementary tables, CSVs, JSON summaries, and LaTeX tables.
- `source_evidence/`
  - Copied source evidence folders from protocol extensions and physiological audits.
- `inventory/`
  - File inventory and missing/skipped file report.

## Important rule

Use this folder for writing and checking the manuscript, but do not delete the original output folders.
"""
    readme.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=str, default=".", help="Project root. Default: current directory.")
    parser.add_argument("--out-dir", type=str, default="outputs/manuscript_locked_results")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite copied files in locked folder.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be copied without copying.")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    out_root = (project_root / args.out_dir).resolve()
    roots = iter_existing_roots(project_root, out_root)

    records: list[CopyRecord] = []

    ensure_dir(out_root)

    # Main paper figures
    copy_named_files(
        roots=roots,
        filenames=MAIN_PAPER_FIGURES,
        dst_dir=out_root / "main_paper" / "figures",
        category="main_paper_figure",
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        records=records,
    )

    # Optional main or supplementary figures
    copy_named_files(
        roots=roots,
        filenames=OPTIONAL_MAIN_OR_SUPPLEMENT_FIGURES,
        dst_dir=out_root / "main_paper" / "optional_figures",
        category="optional_main_or_supplement_figure",
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        records=records,
    )

    # Supplementary figures
    copy_pattern_files(
        roots=roots,
        patterns=SUPPLEMENTARY_FIGURE_PATTERNS,
        dst_dir=out_root / "supplementary" / "figures",
        category="supplementary_figure",
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        records=records,
        allowed_suffixes={".pdf", ".png", ".svg"},
    )

    # Supplementary/source tables and summaries
    copy_pattern_files(
        roots=roots,
        patterns=MAIN_AND_SUPPLEMENT_TABLE_PATTERNS,
        dst_dir=out_root / "supplementary" / "tables",
        category="table_or_summary",
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        records=records,
        allowed_suffixes={".csv", ".json", ".tex", ".md", ".txt"},
    )

    # Copy complete evidence folders in structured form
    copy_source_evidence_folders(project_root, out_root, args.overwrite, args.dry_run, records)

    if not args.dry_run:
        write_readme(out_root)

    write_inventory(records, out_root, args.dry_run)

    copied = sum(r.status in {"copied", "copied_dry_run"} for r in records)
    missing = sum(r.status in {"missing", "missing_folder"} for r in records)
    skipped = sum(r.status == "skipped_exists" for r in records)

    print("\nStage 26A complete.")
    print(f"Project root: {project_root}")
    print(f"Output folder: {out_root}")
    print(f"Copied records: {copied}")
    print(f"Missing records: {missing}")
    print(f"Skipped existing records: {skipped}")
    print("\nInventory:")
    print(out_root / "inventory" / "manuscript_locked_results_inventory.csv")
    print(out_root / "inventory" / "missing_or_skipped_files.csv")

    if missing > 0:
        print("\nWARNING: Some expected files were not found.")
        print("Check the missing report above. This may simply mean a figure was not generated yet.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())