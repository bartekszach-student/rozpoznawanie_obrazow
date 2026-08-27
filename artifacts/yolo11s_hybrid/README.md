# YOLO11s Hybrid — najlepszy zapisany model

Model został wybrany na podstawie końcowej ewaluacji na wydzielonym zbiorze
testowym. Dataset nie jest częścią repozytorium.

## Wyniki testowe

| Metryka | Wynik |
|---|---:|
| Precision | 0.699985 (70,00%) |
| Recall | 0.655917 (65,59%) |
| F1 | 0.677235 (67,72%) |
| mAP@50 | 0.721673 (72,17%) |
| mAP@50-95 | 0.493515 (49,35%) |

## Zawartość

- `weights/best.pt` — najlepszy checkpoint PyTorch (18,33 MB),
- `training/results.csv` — metryki i straty dla każdej epoki,
- `training/results.png` — zbiorczy wykres przebiegu treningu,
- `training/*curve.png` — krzywe Precision, Recall, F1 i PR,
- `training/confusion_matrix*.png` — macierze pomyłek z walidacji,
- `evaluation/metrics_summary.json` — dokładne metryki testowe,
- `evaluation/*curve.png` — krzywe dla zbioru testowego,
- `evaluation/val_batch*_pred.jpg` — przykłady predykcji testowych,
- `training_config.yaml` — najważniejsze parametry treningu.

SHA-256 pliku `best.pt`:

```text
9029E1506CBF4A5F8D381D2EA594F19358DC1FEAB8D03178197947BA283A308D
```

## Uruchomienie ewaluacji

Z katalogu głównego projektu:

```bat
python scripts/evaluate.py --model artifacts/yolo11s_hybrid/weights/best.pt --split test
```
