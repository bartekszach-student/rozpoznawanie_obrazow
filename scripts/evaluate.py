"""Evaluate a trained YOLO ship detector and save metrics and plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


DEFAULT_MODEL = Path(
    "runs/detect/runs/detect/yolo11n_ships/weights/best.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the YOLO ship detector.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=Path("configs/ships.yaml"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", default="runs/evaluation")
    parser.add_argument("--name", default="test_best")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model.resolve()
    data_path = args.data.resolve()

    if not model_path.is_file():
        raise FileNotFoundError(f"Nie znaleziono modelu: {model_path}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Nie znaleziono konfiguracji: {data_path}")

    print(f"Model: {model_path}")
    print(f"Oceniany zbiór: {args.split}")

    # Do oceny uzywamy best.pt, czyli checkpointu o najlepszej walidacji,
    # a niekoniecznie last.pt z ostatniej wykonanej epoki.
    model = YOLO(str(model_path))
    # val() nie zmienia wag. Uruchamia tylko inferencje i porownuje predykcje
    # z etykietami wybranego splitu. plots=True zapisuje krzywe i przyklady.
    metrics = model.val(
        data=str(data_path),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        plots=True,
        project=args.project,
        name=args.name,
        verbose=True,
    )

    # Precision: jaka czesc wykrytych ramek byla poprawna.
    # Recall: jaka czesc wszystkich opisanych statkow zostala odnaleziona.
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    # F1 jest srednia harmoniczna Precision i Recall; mocno karze sytuacje,
    # w ktorej tylko jedna z tych dwoch metryk jest wysoka.
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    summary = {
        "model": str(model_path),
        "split": args.split,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        # mAP50 akceptuje ramke przy IoU >= 0.50. mAP50-95 usrednia wiele
        # coraz ostrzejszych progow, dlatego lepiej ocenia dokladnosc ramki.
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }

    save_dir = Path(metrics.save_dir)
    summary_path = save_dir / "metrics_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== WYNIKI ===")
    print(f"Precision:  {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Recall:     {recall:.4f} ({recall * 100:.2f}%)")
    print(f"F1 score:   {f1:.4f} ({f1 * 100:.2f}%)")
    print(f"mAP@50:     {metrics.box.map50:.4f} ({metrics.box.map50 * 100:.2f}%)")
    print(f"mAP@50-95:  {metrics.box.map:.4f} ({metrics.box.map * 100:.2f}%)")
    print(f"Wykresy i przykłady: {save_dir}")
    print(f"Podsumowanie JSON: {summary_path}")


if __name__ == "__main__":
    main()
