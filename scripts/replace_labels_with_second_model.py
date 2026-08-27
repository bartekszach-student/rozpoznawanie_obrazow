"""Safely replace dataset labels with completed second-model predictions."""

from __future__ import annotations

import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Back up and replace YOLO labels with YOLO11x predictions."
    )
    parser.add_argument("--dataset", type=Path, default=Path("datasets"))
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("second_model_review/predictions"),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test"],
        required=True,
        help="Explicitly select which dataset splits to replace.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform replacement. Without this flag only a dry run is done.",
    )
    return parser.parse_args()


def image_stems(images_dir: Path) -> set[str]:
    return {
        path.stem
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset.resolve()
    predictions_dir = args.predictions.resolve()
    checked: dict[str, tuple[Path, Path, set[str]]] = {}

    for split in args.splits:
        images_dir = dataset_dir / split / "images"
        target_labels_dir = dataset_dir / split / "labels"
        source_labels_dir = predictions_dir / split / "labels"

        if not images_dir.is_dir():
            raise FileNotFoundError(f"Brak folderu obrazow: {images_dir}")
        if not target_labels_dir.is_dir():
            raise FileNotFoundError(f"Brak obecnych etykiet: {target_labels_dir}")
        if not source_labels_dir.is_dir():
            raise FileNotFoundError(f"Brak etykiet YOLO11x: {source_labels_dir}")

        images = image_stems(images_dir)
        predictions = {path.stem for path in source_labels_dir.glob("*.txt")}
        # Nie wolno podmieniac czesci splitu: mieszanka starych i nowych
        # pseudoetykiet utrudnilaby interpretacje kolejnego treningu.
        missing = sorted(images - predictions)
        if missing:
            raise RuntimeError(
                f"{split}: brakuje {len(missing)} etykiet YOLO11x, np. {missing[:5]}"
            )

        checked[split] = (target_labels_dir, source_labels_dir, images)
        print(
            f"{split}: kompletne przewidywania YOLO11x dla {len(images)} obrazow"
        )

    # Domyslnie wykonujemy tylko dry-run. Flaga --apply jest swiadomym
    # potwierdzeniem operacji nadpisujacej etykiety.
    if not args.apply:
        print("\nTRYB TESTOWY: niczego nie zmieniono.")
        print("Aby wykonac kopie i zamiane, dodaj parametr --apply.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = dataset_dir.parent / "backups" / f"labels_before_yolo11x_{timestamp}"

    for split, (target_dir, source_dir, images) in checked.items():
        backup_dir = backup_root / split
        backup_dir.mkdir(parents=True, exist_ok=True)

        # Najpierw pelna kopia poprzednich etykiet, aby eksperyment byl
        # odwracalny i mozna bylo porownac oba warianty zbioru.
        for old_label in target_dir.glob("*.txt"):
            shutil.copy2(old_label, backup_dir / old_label.name)

        for stem in sorted(images):
            source = source_dir / f"{stem}.txt"
            target = target_dir / f"{stem}.txt"
            temporary = target.with_suffix(".txt.yolo11x_tmp")
            shutil.copy2(source, temporary)
            # os.replace wykonuje atomowa podmiane: nie zostawi polowy pliku,
            # nawet gdy operacja zostanie przerwana w trakcie kopiowania.
            os.replace(temporary, target)

        print(f"{split}: zastapiono {len(images)} etykiet")

    print(f"\nKopia poprzednich etykiet: {backup_root}")
    print("Zamiana zakonczona. Przed treningiem uruchom kontrole zbioru.")


if __name__ == "__main__":
    main()
