"""Compare existing YOLO labels with predictions from a stronger second model.

Nothing in the original dataset is overwritten. Predictions and review images are
written to a separate output directory so a person can inspect disagreements.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare current ship labels with predictions from YOLO11x."
    )
    parser.add_argument("--dataset", type=Path, default=Path("datasets"))
    parser.add_argument("--output", type=Path, default=Path("second_model_review"))
    parser.add_argument("--model", default="yolo11x.pt")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--iou-nms", type=float, default=0.60)
    parser.add_argument(
        "--match-iou",
        type=float,
        default=0.50,
        help="Minimum IoU for considering old and new boxes equal.",
    )
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Run inference again even when a second-model label already exists.",
    )
    return parser.parse_args()


def read_yolo_boxes(label_path: Path) -> list[tuple[float, float, float, float]]:
    boxes: list[tuple[float, float, float, float]] = []
    if not label_path.is_file():
        return boxes

    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        x_center, y_center, width, height = map(float, fields[1:5])
        boxes.append((x_center, y_center, width, height))
    return boxes


def xywh_to_xyxy(
    box: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    x_center, y_center, width, height = box
    return (
        x_center - width / 2.0,
        y_center - height / 2.0,
        x_center + width / 2.0,
        y_center + height / 2.0,
    )


def box_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    """Oblicz IoU, czyli czesc wspolna ramek podzielona przez ich sume.

    IoU bliskie 1 oznacza prawie identyczne ramki, a IoU 0 brak przeciecia.
    """
    first_xyxy = xywh_to_xyxy(first)
    second_xyxy = xywh_to_xyxy(second)
    intersection_width = max(
        0.0, min(first_xyxy[2], second_xyxy[2]) - max(first_xyxy[0], second_xyxy[0])
    )
    intersection_height = max(
        0.0, min(first_xyxy[3], second_xyxy[3]) - max(first_xyxy[1], second_xyxy[1])
    )
    intersection = intersection_width * intersection_height
    first_area = max(0.0, first_xyxy[2] - first_xyxy[0]) * max(
        0.0, first_xyxy[3] - first_xyxy[1]
    )
    second_area = max(0.0, second_xyxy[2] - second_xyxy[0]) * max(
        0.0, second_xyxy[3] - second_xyxy[1]
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def matched_box_count(
    old_boxes: list[tuple[float, float, float, float]],
    new_boxes: list[tuple[float, float, float, float]],
    threshold: float,
) -> int:
    """Polacz stare i nowe ramki bez przypisywania jednej ramki dwa razy."""
    candidates: list[tuple[float, int, int]] = []
    for old_index, old_box in enumerate(old_boxes):
        for new_index, new_box in enumerate(new_boxes):
            iou = box_iou(old_box, new_box)
            if iou >= threshold:
                candidates.append((iou, old_index, new_index))

    used_old: set[int] = set()
    used_new: set[int] = set()
    matches = 0
    for _, old_index, new_index in sorted(candidates, reverse=True):
        if old_index in used_old or new_index in used_new:
            continue
        used_old.add(old_index)
        used_new.add(new_index)
        matches += 1
    return matches


def classify_comparison(old_count: int, new_count: int, matches: int) -> str:
    """Podziel obrazy na zgodne i wymagajace kontroli czlowieka."""
    if old_count == 0 and new_count == 0:
        return "agreement_empty"
    if old_count == 0 and new_count > 0:
        return "empty_but_detected"
    if old_count > 0 and new_count == 0:
        return "existing_but_missed"
    if old_count == new_count == matches:
        return "agreement"
    return "model_disagreement"


def save_yolo_predictions(
    label_path: Path,
    boxes: list[tuple[float, float, float, float]],
) -> None:
    lines = [
        f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        for x_center, y_center, width, height in boxes
    ]
    contents = "\n".join(lines)
    if contents:
        contents += "\n"
    label_path.write_text(contents, encoding="utf-8")


def draw_boxes(
    image_path: Path,
    old_boxes: list[tuple[float, float, float, float]],
    new_boxes: list[tuple[float, float, float, float]],
    output_path: Path,
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        return
    image_height, image_width = image.shape[:2]

    def draw(
        boxes: list[tuple[float, float, float, float]],
        color: tuple[int, int, int],
        label: str,
    ) -> None:
        for box in boxes:
            x1, y1, x2, y2 = xywh_to_xyxy(box)
            point1 = (int(x1 * image_width), int(y1 * image_height))
            point2 = (int(x2 * image_width), int(y2 * image_height))
            cv2.rectangle(image, point1, point2, color, 2)
            cv2.putText(
                image,
                label,
                (point1[0], max(18, point1[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

    draw(old_boxes, (0, 255, 0), "OLD")
    draw(new_boxes, (0, 0, 255), "YOLO11x")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "comparison_report.csv"
    # YOLO11x jest tu nauczycielem, a nie modelem docelowym dla Jetsona.
    # Jego propozycje sa silniejsze, ale nadal nie sa automatycznie prawda.
    model = YOLO(args.model)
    totals: Counter[str] = Counter()

    with report_path.open("w", newline="", encoding="utf-8") as report_file:
        writer = csv.writer(report_file)
        writer.writerow(
            ["split", "image", "category", "old_boxes", "new_boxes", "matches"]
        )

        for split in args.splits:
            images_dir = dataset_dir / split / "images"
            old_labels_dir = dataset_dir / split / "labels"
            new_labels_dir = output_dir / "predictions" / split / "labels"
            new_labels_dir.mkdir(parents=True, exist_ok=True)

            if not images_dir.is_dir():
                print(f"Pomijam {split}: brak folderu {images_dir}")
                continue

            images = sorted(
                path
                for path in images_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            if args.max_images is not None:
                images = images[: args.max_images]

            print(f"{split}: porownywanie {len(images)} obrazow...")
            for image_number, image_path in enumerate(images, start=1):
                old_label_path = old_labels_dir / f"{image_path.stem}.txt"
                new_label_path = new_labels_dir / f"{image_path.stem}.txt"
                old_boxes = read_yolo_boxes(old_label_path)

                # Zapisane predykcje dzialaja jak cache i pozwalaja wznowic
                # analize po Ctrl+C bez ponownego liczenia gotowych obrazow.
                if new_label_path.exists() and not args.overwrite:
                    new_boxes = read_yolo_boxes(new_label_path)
                else:
                    result = model.predict(
                        source=str(image_path),
                        classes=[8],  # COCO class 8 = boat
                        # Niski prog celowo tworzy wiecej kandydatow. Obrazy,
                        # na ktorych modele sie roznia, trafiaja do review_queue.
                        conf=args.confidence,
                        iou=args.iou_nms,
                        imgsz=args.imgsz,
                        device=args.device,
                        batch=1,
                        half=args.device.lower() != "cpu",
                        verbose=False,
                    )[0]
                    new_boxes = (
                        [tuple(box) for box in result.boxes.xywhn.cpu().tolist()]
                        if result.boxes is not None
                        else []
                    )
                    save_yolo_predictions(new_label_path, new_boxes)

                matches = matched_box_count(old_boxes, new_boxes, args.match_iou)
                category = classify_comparison(len(old_boxes), len(new_boxes), matches)
                totals[category] += 1
                writer.writerow(
                    [split, image_path.name, category, len(old_boxes), len(new_boxes), matches]
                )

                # Nie kopiujemy tysiecy oczywistych przypadkow. Wizualizacje
                # zapisujemy tylko dla rozbieznosci wymagajacych sprawdzenia.
                if category not in {"agreement", "agreement_empty"}:
                    review_path = output_dir / "review_queue" / category / split / image_path.name
                    draw_boxes(image_path, old_boxes, new_boxes, review_path)

                if image_number % 50 == 0 or image_number == len(images):
                    print(f"{split}: {image_number}/{len(images)}")

    print("\n=== PODSUMOWANIE ===")
    for category, count in sorted(totals.items()):
        print(f"{category}: {count}")
    print(f"Raport: {report_path}")
    print(f"Obrazy do kontroli: {output_dir / 'review_queue'}")
    print("Oryginalne etykiety nie zostaly zmienione.")


if __name__ == "__main__":
    main()
