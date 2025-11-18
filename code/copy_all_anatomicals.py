#!/usr/bin/env python3
"""
Copy all subject/session anat folders into a single Dropbox directory.

Each source dataset lives under a directory named like:
    resampled_bold_sub-BOYER_ses-ACTIVE--2024-09-27
and contains an anat folder at:
    input/<flywheel_id>/sub-BOYER/ses-ACTIVE/anat

We copy every anat folder into <dest>/anat_BOYER_ACTIVE.
"""

import argparse
import re
import shutil
from pathlib import Path
from typing import Iterable, NamedTuple, Optional


class Dataset(NamedTuple):
    subject: str
    session: str
    anat_path: Path


RESAMPLED_DIR_REGEX = re.compile(
    r"^resampled_bold_sub-(?P<subject>[A-Za-z0-9]+)_ses-(?P<session>[A-Za-z0-9]+)--"
)


def discover_datasets(source_root: Path) -> Iterable[Dataset]:
    """Yield Dataset entries for every resampled directory that has an anat folder."""
    for dataset_dir in sorted(source_root.glob("resampled_bold_sub-*")):
        if not dataset_dir.is_dir():
            continue
        match = RESAMPLED_DIR_REGEX.match(dataset_dir.name)
        if not match:
            print(f"[WARN] Skipping {dataset_dir.name}: does not match expected pattern")
            continue

        anat_path = _resolve_anat_path(dataset_dir)
        if anat_path is None:
            print(f"[WARN] Skipping {dataset_dir.name}: could not find anat folder")
            continue

        yield Dataset(
            subject=match.group("subject").upper(),
            session=match.group("session").upper(),
            anat_path=anat_path,
        )


def _resolve_anat_path(dataset_dir: Path) -> Optional[Path]:
    """Return the inner anat folder for a dataset directory, or None if missing."""
    candidates = sorted(dataset_dir.glob("input/*/sub-*/ses-*/anat"))
    if not candidates:
        return None
    if len(candidates) > 1:
        print(f"[INFO] Multiple anat folders in {dataset_dir.name}; using {candidates[-1]}")
    return candidates[-1]


def copy_anat_folders(source_root: Path, dest_root: Path, overwrite: bool) -> None:
    """Copy every discovered anat folder into dest_root."""
    dest_root.mkdir(parents=True, exist_ok=True)
    copied, skipped = 0, 0

    for dataset in discover_datasets(source_root):
        dest_dir = dest_root / f"anat_{dataset.subject}_{dataset.session}"
        if dest_dir.exists() and not overwrite:
            print(f"[SKIP] {dest_dir} exists (use --overwrite to replace)")
            skipped += 1
            continue

        if dest_dir.exists() and overwrite:
            shutil.rmtree(dest_dir)

        print(f"[COPY] {dataset.anat_path} -> {dest_dir}")
        shutil.copytree(dataset.anat_path, dest_dir, dirs_exist_ok=False)
        copied += 1

    print(f"Done. Copied: {copied}, skipped: {skipped}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help=(
            "Directory that contains resampled_bold_sub-* folders "
            "(e.g., /Volumes/JaceksMacStudio/jacekdmochowski/PROJECTS/fus/data/resampled_bold_flywheel)"
        ),
    )
    parser.add_argument(
        "--dest-root",
        type=Path,
        default=Path(
            "/Users/jacekdmochowski/City College Dropbox/Jacek Dmochowski/sharing/fus_bold/all_anatomicals"
        ),
        help="Destination directory where anat_SUBJECT_SESSION folders will be created.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace destination folders if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    copy_anat_folders(args.source_root.expanduser(), args.dest_root.expanduser(), args.overwrite)


if __name__ == "__main__":
    main()
