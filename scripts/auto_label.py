"""Generate initial YOLO labels for ships with a pretrained COCO model.

The COCO "boat" class (id 8) is converted to the local "ship" class (id 0).
The generated labels are pseudo-labels and must be reviewed by a person.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create YOLO pseudo-labels for ships in train/val/test folders."
    )
    parser.add_argument("--dataset", type=Path, default=Path("datasets"))
    parser.add_argument("--model", default="yolo11m.pt")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--confidence", type=float, default=0.20)
    parser.add_argument("--iou", type=float, default=0.60)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="0", help="GPU number, e.g. 0, or cpu")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limit images per split for a test run.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace labels that already exist.",
    )
    parser.add_argument(
        "--save-previews",
        action="store_true",
        help="Save images with predicted boxes in <split>/previews.",
    )
    return parser.parse_args()


def find_images(images_dir: Path, limit: int | None) -> list[Path]:
    """Zbierz obrazy w stalej kolejnosci, aby eksperyment byl powtarzalny."""
    images = sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    return images[:limit] if limit is not None else images


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset.resolve()
    model = YOLO(args.model)
    report_path = dataset_dir / "pseudo_labels_report.csv"

    total_images = 0
    total_boxes = 0
    skipped = 0

    with report_path.open("w", newline="", encoding="utf-8") as report_file:
        writer = csv.writer(report_file)
        writer.writerow(["split", "image", "status", "detections", "max_confidence"])

        for split in args.splits:
            images_dir = dataset_dir / split / "images"
            labels_dir = dataset_dir / split / "labels"
            previews_dir = dataset_dir / split / "previews"

            if not images_dir.is_dir():
                print(f"Pomijam {split}: brak folderu {images_dir}")
                continue

            labels_dir.mkdir(parents=True, exist_ok=True)
            if args.save_previews:
                previews_dir.mkdir(parents=True, exist_ok=True)

            images = find_images(images_dir, args.max_images)
            pending_images: list[Path] = []

            for image_path in images:
                label_path = labels_dir / f"{image_path.stem}.txt"
                # Nie nadpisujemy istniejacych etykiet. Dzieki temu przerwany
                # proces mozna bezpiecznie wznowic i zachowac reczne poprawki.
                if label_path.exists() and not args.overwrite:
                    writer.writerow([split, image_path.name, "skipped_existing", "", ""])
                    skipped += 1
                else:
                    pending_images.append(image_path)

            if not pending_images:
                print(f"{split}: brak nowych obrazow do opisania")
                continue

            print(f"{split}: automatyczne opisywanie {len(pending_images)} obrazow...")
            # Przetwarzamy jeden obraz naraz. Podanie tysięcy sciezek w jednym
            # wywolaniu utworzyloby ogromny tensor i zapelnilo 8 GB VRAM.
            for image_number, image_path in enumerate(pending_images, start=1):
                result = model.predict(
                    source=str(image_path),
                    # W modelach COCO identyfikator 8 oznacza klase "boat".
                    # Po zapisie zmieniamy ja na lokalna klase 0 = "ship".
                    classes=[8],  # COCO class 8 = boat
                    # Nizszy prog zwieksza Recall (mniej pominietych statkow),
                    # ale tworzy wiecej falszywych propozycji do sprawdzenia.
                    conf=args.confidence,
                    iou=args.iou,
                    imgsz=args.imgsz,
                    device=args.device,
                    batch=1,
                    half=args.device.lower() != "cpu",
                    verbose=False,
                )[0]

                label_path = labels_dir / f"{image_path.stem}.txt"
                lines: list[str] = []
                confidences: list[float] = []

                if result.boxes is not None:
                    # xywhn to srodek ramki, szerokosc i wysokosc znormalizowane
                    # do zakresu 0-1, czyli dokladnie format wymagany przez YOLO.
                    coordinates = result.boxes.xywhn.cpu().tolist()
                    confidences = result.boxes.conf.cpu().tolist()
                    for x_center, y_center, width, height in coordinates:
                        lines.append(
                            f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                        )

                contents = "\n".join(lines)
                if contents:
                    contents += "\n"
                # Brak detekcji tworzy pusty plik TXT. Dla YOLO oznacza to
                # poprawny przyklad tla, a nie brakujaca etykiete.
                label_path.write_text(contents, encoding="utf-8")

                if args.save_previews:
                    import cv2

                    cv2.imwrite(str(previews_dir / image_path.name), result.plot())

                detection_count = len(lines)
                max_confidence = max(confidences) if confidences else 0.0
                writer.writerow(
                    [
                        split,
                        image_path.name,
                        "generated",
                        detection_count,
                        f"{max_confidence:.4f}",
                    ]
                )
                total_images += 1
                total_boxes += detection_count

                if image_number % 50 == 0 or image_number == len(pending_images):
                    print(
                        f"{split}: {image_number}/{len(pending_images)} obrazow, "
                        f"ramki w tej sesji: {total_boxes}"
                    )

    print("\nGotowe.")
    print(f"Opisane obrazy: {total_images}")
    print(f"Utworzone ramki: {total_boxes}")
    print(f"Pominiete istniejace etykiety: {skipped}")
    print(f"Raport: {report_path}")
    print("UWAGA: pseudoetykiety trzeba sprawdzic i poprawic przed treningiem.")


if __name__ == "__main__":
    main()
