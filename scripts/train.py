"""Validate the ship dataset and train a YOLO detector."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO11 on the ship dataset.")
    # Parametry CLI pozwalaja powtarzac eksperymenty bez edytowania kodu.
    # Kazdy trening zapisuje args.yaml, wiec pozniej wiadomo, jak powstal model.
    parser.add_argument("--data", type=Path, default=Path("configs/ships.yaml"))
    # yolo11n.pt zawiera wagi wyuczone wczesniej na COCO. Zaczynamy od nich
    # (transfer learning), zamiast uczyc cala siec od losowych wag.
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument(
        "--batch",
        type=int,
        default=-1,
        help="-1 lets Ultralytics select a batch for about 60%% GPU memory usage.",
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="yolo11n_ships")
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to last.pt from an interrupted training run.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate the dataset without starting training.",
    )
    return parser.parse_args()


def resolve_dataset_root(config_path: Path, config: dict) -> Path:
    root = Path(config.get("path", config_path.parent))
    if not root.is_absolute():
        root = (config_path.parent / root).resolve()
    return root


def validate_dataset(config_path: Path) -> None:
    """Sprawdz dane zanim GPU zacznie kosztowny trening.

    YOLO wymaga pliku TXT o tej samej nazwie dla kazdego obrazu. Pusty plik
    jest poprawny i oznacza obraz tla, na ktorym nie ma obiektu klasy ship.
    """
    if not config_path.is_file():
        raise FileNotFoundError(f"Nie znaleziono konfiguracji: {config_path}")

    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError("Plik YAML nie zawiera poprawnej konfiguracji.")

    class_names = config.get("names", {})
    if isinstance(class_names, list):
        valid_classes = set(range(len(class_names)))
    elif isinstance(class_names, dict):
        valid_classes = {int(class_id) for class_id in class_names}
    else:
        raise ValueError("Pole 'names' w YAML musi być listą albo słownikiem.")

    dataset_root = resolve_dataset_root(config_path, config)
    errors: list[str] = []
    warnings: list[str] = []

    for split in ("train", "val", "test"):
        split_value = config.get(split)
        if split_value is None:
            if split != "test":
                errors.append(f"Brak pola '{split}' w konfiguracji.")
            continue

        images_dir = Path(split_value)
        if not images_dir.is_absolute():
            images_dir = dataset_root / images_dir
        labels_dir = images_dir.parent / "labels"

        if not images_dir.is_dir():
            errors.append(f"{split}: brak folderu obrazów {images_dir}")
            continue
        if not labels_dir.is_dir():
            errors.append(f"{split}: brak folderu etykiet {labels_dir}")
            continue

        images = {
            path.stem: path
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        }
        labels = {path.stem: path for path in labels_dir.glob("*.txt")}
        missing_labels = sorted(images.keys() - labels.keys())
        orphan_labels = sorted(labels.keys() - images.keys())

        if missing_labels:
            errors.append(
                f"{split}: {len(missing_labels)} obrazów nie ma etykiety, "
                f"np. {missing_labels[:3]}"
            )
        if orphan_labels:
            warnings.append(
                f"{split}: {len(orphan_labels)} etykiet nie ma obrazu, "
                f"np. {orphan_labels[:3]}"
            )

        # Puste etykiety sa potrzebne: ucza model, jak wyglada woda, brzeg lub
        # port bez statku i pomagaja ograniczac falszywe alarmy.
        empty_labels = 0
        box_count = 0
        for stem in sorted(images.keys() & labels.keys()):
            label_path = labels[stem]
            lines = [
                line.strip()
                for line in label_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if not lines:
                empty_labels += 1
                continue

            for line_number, line in enumerate(lines, start=1):
                fields = line.split()
                if len(fields) != 5:
                    errors.append(
                        f"{label_path}:{line_number}: oczekiwano 5 wartości, "
                        f"otrzymano {len(fields)}"
                    )
                    continue
                try:
                    class_id = int(fields[0])
                    x_center, y_center, width, height = map(float, fields[1:])
                except ValueError:
                    errors.append(f"{label_path}:{line_number}: niepoprawne liczby")
                    continue

                if class_id not in valid_classes:
                    errors.append(
                        f"{label_path}:{line_number}: nieznana klasa {class_id}"
                    )
                if not all(0.0 <= value <= 1.0 for value in (x_center, y_center, width, height)):
                    errors.append(
                        f"{label_path}:{line_number}: współrzędne muszą być w zakresie 0-1"
                    )
                if width <= 0.0 or height <= 0.0:
                    errors.append(
                        f"{label_path}:{line_number}: szerokość i wysokość muszą być dodatnie"
                    )
                box_count += 1

        print(
            f"{split}: obrazy={len(images)}, etykiety={len(labels)}, "
            f"ramki={box_count}, puste={empty_labels}"
        )

    for warning in warnings:
        print(f"OSTRZEŻENIE: {warning}")

    if errors:
        preview = "\n".join(f"- {error}" for error in errors[:20])
        extra = "" if len(errors) <= 20 else f"\n...oraz {len(errors) - 20} kolejnych"
        raise ValueError(f"Zbiór zawiera {len(errors)} błędów:\n{preview}{extra}")

    print("Kontrola zbioru zakończona pomyślnie.")


def main() -> None:
    args = parse_args()
    config_path = args.data.resolve()
    validate_dataset(config_path)

    if args.check_only:
        return

    if args.device.lower() != "cpu":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA nie jest dostępna. Sprawdź środowisko PyTorch albo użyj --device cpu."
            )
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if args.resume is not None:
        checkpoint = args.resume.resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Nie znaleziono checkpointu: {checkpoint}")
        print(f"Wznawianie treningu z: {checkpoint}")
        # resume=True odtwarza nie tylko wagi, ale tez epoke, optymalizator
        # i harmonogram learning rate zapisane w last.pt.
        YOLO(str(checkpoint)).train(resume=True)
        return

    print(f"Model startowy: {args.model}")
    # Zaladowanie pliku .pt uruchamia transfer learning. Ostatnia warstwa
    # detekcyjna zostanie dostosowana do jednej klasy zdefiniowanej w YAML.
    model = YOLO(args.model)
    model.train(
        data=str(config_path),
        # Epoka to jedno przejscie przez caly zbior train. EarlyStopping moze
        # zakonczyc trening wczesniej, gdy walidacja przestanie sie poprawiac.
        epochs=args.epochs,
        # Obrazy sa skalowane do imgsz. Wieksza wartosc pomaga przy malych,
        # odleglych statkach, ale zwieksza zuzycie VRAM i czas obliczen.
        imgsz=args.imgsz,
        # batch=-1 zleca Ultralytics automatyczny dobor batcha do pamieci GPU.
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        # patience okresla liczbe epok bez poprawy przed EarlyStopping.
        patience=args.patience,
        project=args.project,
        name=args.name,
        pretrained=True,
        # AMP uzywa mieszanej precyzji FP16/FP32: zmniejsza zuzycie VRAM
        # i zwykle przyspiesza trening na karcie NVIDIA.
        amp=True,
        # Staly seed ulatwia uczciwe porownywanie kolejnych eksperymentow.
        seed=42,
        deterministic=True,
        # plots=True zapisuje m.in. krzywe strat, PR i macierz pomylek.
        plots=True,
        # Poniewaz nie podajemy tu parametrow hsv_*, scale, mosaic itd.,
        # Ultralytics stosuje swoje domyslne augmentacje tylko dla train.
        # Zbior val nie jest augmentowany i sluzy do wyboru best.pt.
    )


if __name__ == "__main__":
    main()
