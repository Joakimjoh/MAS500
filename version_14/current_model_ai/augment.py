# Run this at the start
import xml.etree.ElementTree as ET
from ultralytics import YOLO
from pathlib import Path
import random
import shutil
import cv2
import itertools
import os
import shutil
from ultralytics import YOLO
import matplotlib.pyplot as plt
import albumentations as A

data_yaml = '/home/student/Desktop/current_model_ai/data.yaml'
project_root = '/home/student/Desktop/current_model_ai/training_results'

# Clean project folder
if os.path.exists(project_root):
    shutil.rmtree(project_root)

# Phase 1: Model selection
models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']
best_model = ['yolov8s.pt']
model_scores = {}

# Pick best epoch/batch combo
best_epochs, best_batch = 30, 8
print(f"\n✅ Best config: epochs={best_epochs}, batch={best_batch}")

# Phase 3: Grid Search on augmentations
shear_vals = [0.0, 5.0]
scale_vals = [0.05, 0.1]
translate_vals = [0.05, 0.1]
degrees_vals = [0.0, 10.0, 15.0]
param_grid = list(itertools.product(shear_vals, scale_vals, translate_vals, degrees_vals))
grid_results = []

print("\n🔧 Phase 3: Augmentation grid search\n")

for i, (shear, scale, translate, degrees) in enumerate(param_grid):
    run_name = f'grid_shear{shear}_scale{scale}_trans{translate}_deg{degrees}'
    print(f"Training {run_name} ({i+1}/{len(param_grid)})...")
    model = YOLO(best_model)
    model.train(
        data=data_yaml,
        epochs=best_epochs,
        batch=best_batch,
        imgsz=640,
        name=run_name,
        project=project_root,
        shear=shear,
        scale=scale,
        translate=translate,
        degrees=degrees,
        augment=True,
        patience=10,
        verbose=False
    )
    score = model.val(data=data_yaml).box.map50
    grid_results.append((run_name, score))
    print(f"📊 {run_name} → mAP50 = {score:.3f}")

# Show top 5 grid results
grid_results.sort(key=lambda x: x[1], reverse=True)
print("\n🏁 Top 5 Grid Search Results:")
for name, score in grid_results[:5]:
    print(f"{name}: mAP50 = {score:.3f}")
