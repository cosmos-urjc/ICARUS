#!/usr/bin/env python3
"""
Validate several YOLO models on multiple datasets and build a leaderboard.

- Each model is evaluated on every dataset YAML you list.
- Metrics per‑dataset are recorded, then the script averages them per model.
- A sorted table (by mAP50‑95) is printed, plus CSV exports for full and
  averaged results.

Requires: ultralytics >= 8.1.0, torch, pandas
"""

from ultralytics import YOLO
import torch
import pandas as pd
import gc

# ------------------------------------------------------------------
# 1) Put here the models (weights) you want to compare
# ------------------------------------------------------------------
MODELS = [
    "yolo11m.pt",
    "yolo11l.pt",
    "yolo11x.pt",
    "yolov8n.pt",
    "yolo11n.pt",
    "runs/drone/yolov8n_quick/weights/best.pt",
    "runs/drone/yolov8n_full/weights/best.pt",
    "runs/drone/yolo11n_quick/weights/best.pt",
    "runs/drone/yolo11n_full/weights/best.pt",
    "runs/drone/yolo11m_quick/weights/best.pt",
    "runs/test/yolo11n_quick_dataset_full/weights/best.pt"
]

# ------------------------------------------------------------------
# 2) List of datasets: (nice‑name, data.yaml path)
#    Add as many as you need; the script will loop over all.
# ------------------------------------------------------------------
DATASETS = [
    ("full_3", "dataset/dataset_full/data.yaml"),
]

# Device where the inference will run
DEVICE = "cuda:0"        # Cambia a "cpu" si no tienes GPU

# Metric used to sort the final leaderboard
METRIC_MAIN = "mAP50-95"

# ------------------------------------------------------------------
# 3) Evaluation loop: model × dataset
# ------------------------------------------------------------------
results = []

for model_path in MODELS:
    print(f"\n======================\nValidating model: {model_path}\n")
    model = YOLO(model_path).to(DEVICE)

    for ds_name, ds_yaml in DATASETS:
        print(f"  -> Dataset: {ds_name}")
        metrics = model.val(data=ds_yaml, device=DEVICE, verbose=False)

        results.append({
            "Model":      model_path,
            "Dataset":    ds_name,
            "mAP50-95":   metrics.box.map,
            "mAP50":      metrics.box.map50,
            "Precision":  metrics.box.mp,
            "Recall":     metrics.box.mr,
        })

    # Liberar memoria GPU después de validar el modelo completo
    del model
    torch.cuda.empty_cache()
    gc.collect()

# ------------------------------------------------------------------
# 4) Construir tablas de resultados
# ------------------------------------------------------------------

df = pd.DataFrame(results)

# Promedio por modelo (across datasets)
avg_df = (df.groupby("Model")[["mAP50-95", "mAP50", "Precision", "Recall"]]
            .mean()
            .reset_index()
            .sort_values(METRIC_MAIN, ascending=False))

print("\n============== Leaderboard (Average Across Datasets) ==============")
print(avg_df.to_string(index=False, float_format="%.3f"))

best_row = avg_df.iloc[0]
print(f"\n🏆 Best model by {METRIC_MAIN}: {best_row.Model} ({best_row[METRIC_MAIN]:.3f})")
