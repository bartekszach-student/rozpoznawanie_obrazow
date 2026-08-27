# Ship Detection

Projekt detekcji statków na obrazach z użyciem modeli YOLO z biblioteki
Ultralytics. Repozytorium zawiera skrypty do automatycznego etykietowania,
kontroli datasetu, treningu i ewaluacji modelu.

## Wymagania

- Python 3.10 lub 3.11
- środowisko Conda `ships`
- karta NVIDIA i zgodna wersja PyTorch/CUDA (opcjonalnie, ale zalecane do treningu)

Instalacja zależności w terminalu Anaconda Prompt:

```bat
conda activate ships
pip install -r requirements.txt
```

## Dataset

Dataset nie jest przechowywany w repozytorium. Lokalnie powinien mieć strukturę:

```text
datasets/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Każdy obraz powinien mieć odpowiadający mu plik `.txt` w formacie YOLO.
Pusty plik etykiety oznacza obraz bez statku. Ścieżkę datasetu ustawia się w
`configs/ships.yaml`.

## Kontrola danych

```bat
python scripts/train.py --check-only
```

## Trening

Przykład treningu YOLO11n przez maksymalnie 80 epok:

```bat
python scripts/train.py --model yolo11n.pt --epochs 80 --imgsz 960 --batch -1
```

Najlepszy checkpoint jest zapisywany w katalogu `runs/` jako `best.pt`.
Katalog wyników i pliki wag są ignorowane przez Git, ponieważ są duże i są
artefaktami eksperymentów.

## Ewaluacja

Podaj rzeczywistą ścieżkę do najlepszego checkpointu:

```bat
python scripts/evaluate.py --model runs/detect/yolo11n_ships/weights/best.pt --split test
```

Skrypt wypisuje Precision, Recall, F1, mAP@50 i mAP@50-95 oraz zapisuje
wykresy ewaluacji w katalogu `runs/evaluation/`.

## Najlepszy zapisany model

Wybrany checkpoint `yolo11s_hybrid` oraz komplet wykresów i wyników znajduje
się w katalogu [`artifacts/yolo11s_hybrid`](artifacts/yolo11s_hybrid).

Wyniki na zbiorze testowym:

- Precision: 70,00%
- Recall: 65,59%
- F1: 67,72%
- mAP@50: 72,17%
- mAP@50-95: 49,35%

Przykładowe użycie opublikowanych wag:

```bat
python scripts/evaluate.py --model artifacts/yolo11s_hybrid/weights/best.pt --split test
```
Uruchomienie sieci z kamery
```bat
yolo predict model="ścieżka\do\best.pt" source=0 imgsz=960 conf=0.34 device=0 show=True
```

## Jetson Xavier i ROS

Eksport oraz integracja z kamerą/ROS są kolejnym etapem projektu. Na Jetsonie
PyTorch i TensorRT muszą być zgodne z zainstalowaną wersją JetPack; dlatego
środowisko Jetsona należy przygotować osobno od środowiska Windows.
