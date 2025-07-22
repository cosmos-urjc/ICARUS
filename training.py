#!/usr/bin/env python3
"""
Refine an existing YOLO checkpoint on the custom drone dataset.

Every argument mirrors the CLI flags you’d pass to:
    yolo detect train ...
so switching between script and terminal is loss-less.
"""

from ultralytics import YOLO   # high-level API for training / val / predict
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# CONFIG – just edit & save
# ──────────────────────────────────────────────────────────────
PRESET = "quick"        # "quick"  = 5-epoch screen,   640 px
#PRESET = "medio"        # "full"   = 100-epoch polish, 960 px

MODELS = [
    #"yolov8n.pt",
    "yolo11n.pt",
    #"yolo11m.pt",
    #"yolo11x.pt",
    #"yolo11l.pt"
]

DEVICE = "0"            # "0" = GPU-0, "cpu" = CPU-only

DATA_YAML = "dataset/dataset_full_3/data.yaml"
#DATA_YAML = "dataset/dataset1/data.yaml"
PROJECT    = "runs/test"

# Preset hyper-parameters
PRESETS = dict(
    quick=dict(
        # —— Dataset paths & schedule ——————————————————————————————
        epochs=2, 
        imgsz=640, 
        #batch=16,
        batch  = -1,        # ← auto-search

        # —— Data augmentation probabilities ————————————————
        mosaic=1.0, 
        mixup=0.0, 
        copy_paste=0.0, 
        multi_scale=False,
        
        # —— Optimiser & LR schedule ——————————————————————————————
        # its on auto so it auto chooses lr0 and lrf
        lr0=0.01, 
        lrf=0.01,
        
        # —— Hardware & performance ————————————————————————————————
        workers     = 12,                     # CPU dataloader subprocesses that prepare the next batch
        #cache       = "disk",                 # hold images/labels in system RAM after first load
        amp         = True,                  # Automatic Mixed Precision (FP16) for faster/leaner training

        # —— Logging / output folders ————————————————————————————
        name_tag="_quick_dataset_full"
    ),
    medio=dict(
        # —— Dataset paths & schedule ——————————————————————————————
        epochs=10, 
        imgsz=832, 
        batch  = -1,        # ← auto-search

        # —— Data augmentation probabilities ————————————————
        mosaic=0.5, 
        mixup=0.0, 
        copy_paste=0.2, 
        multi_scale=True,
        
        # —— Optimiser & LR schedule ——————————————————————————————
        lr0=0.01, 
        lrf=0.01,
        
        # —— Hardware & performance ————————————————————————————————
        workers     = 8,                     # CPU dataloader subprocesses that prepare the next batch
        cache       = "disk",                 # hold images/labels in system RAM after first load
        amp         = True,                  # Automatic Mixed Precision (FP16) for faster/leaner training

        # —— Logging / output folders ————————————————————————————
        name_tag="_medio"
    ),
    full=dict(
        # —— Dataset paths & schedule ——————————————————————————————
        epochs      = 100,                   # number of full passes over the train split
        imgsz       = 832,                   # resize shorter edge → 960 px (keeps aspect after aug)
        #batch       = 8,                     # images per GPU before gradient update
        batch  = -1,        # ← auto-search

        # —— Data augmentation probabilities ————————————————
        mosaic      = 0.5,                   # 50 % chance of 4-image Mosaic each batch
        mixup       = 0.0,                   # 0 % MixUp (disabled)
        copy_paste  = 0.2,                   # 20 % Copy-Paste augmentation
        multi_scale = True,                  # random scale between 0.5–1.5× img size each batch

        # —— Optimiser & LR schedule ——————————————————————————————
        lr0         = 0.005,                 # initial learning-rate at epoch 0
        lrf         = 0.1,                   # final LR = lr0 × lrf (cosine decay curve)

        # —— Hardware & performance ————————————————————————————————
        workers     = 12,                     # CPU dataloader subprocesses that prepare the next batch
        cache       = "disk",                 # hold images/labels in system RAM after first load
        amp         = True,                  # Automatic Mixed Precision (FP16) for faster/leaner training

        # —— Logging / output folders ————————————————————————————
        name_tag="_full_freeze",

        # -- Estra de testeos
        # no funciona - no usar -> demasiados pocos epocks y dataset pequeño
        #freeze=2
    )
)

cfg = PRESETS[PRESET]




# ──────────────────────────────────────────────────────────────
# TRAIN LOOP
# ──────────────────────────────────────────────────────────────
try:

    def train_one(model_path: Path, cfg: dict):
        run_name = f"{model_path.stem}{cfg['name_tag']}"
        print(f"\n🚀  Training {run_name}  ({PRESET})")

        model = YOLO(str(model_path)).to("cuda:"+DEVICE if DEVICE!="cpu" else "cpu")

        # remove helper key before passing
        train_args = {k: v for k, v in cfg.items() if k != "name_tag"}
        metrics = model.train(data=DATA_YAML, project=PROJECT,
                            name=run_name, device=DEVICE, **train_args)

        pre, inf = metrics.speed["preprocess"], metrics.speed["inference"]
        if pre > inf:                                       # CPU bottleneck → double workers once
            new_workers = cfg["workers"] * 2
            print(f"🔄  Preprocess {pre:.2f} ms > inference {inf:.2f} ms."
                f"  Restarting with workers={new_workers} …")
            cfg["workers"] = new_workers
            run_name += f"_w{new_workers}"
            metrics = model.train(data=DATA_YAML, project=PROJECT,
                                name=run_name, device=DEVICE, **train_args)

        map_5095 = metrics.box.map           # overall mAP averaged over IoU 0.50:0.95
        print(f"✅  {run_name}: mAP50-95 = {map_5095:.3f}")

    for mp in MODELS:
        train_one(Path(mp), cfg.copy())

    print("\n🎉 All trainings finished.")

except KeyboardInterrupt:
    print("\n The program was stopped by the user.")