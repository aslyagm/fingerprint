"""Phase 3: run preprocessing on every image and store per-image quality data.

Usage:  python src/phase3_stats.py [data_root]
Writes results/phase3/quality.csv
"""
import glob
import os
import sys

import cv2
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import load, process, quality_features, right_edge_artifact_width  # noqa: E402

root = sys.argv[1] if len(sys.argv) > 1 else "."
out_dir = os.path.join("results", "phase3")
os.makedirs(out_dir, exist_ok=True)

rows = []
files = sorted(glob.glob(os.path.join(root, "Db1_[ab]", "*.tif")))
for i, f in enumerate(files):
    raw = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    finger, imp = os.path.splitext(os.path.basename(f))[0].split("_")
    q = quality_features(process(load(f)))
    q.update(file=os.path.relpath(f, root).replace("\\", "/"),
             subset=os.path.basename(os.path.dirname(f)),
             finger=int(finger), impression=int(imp),
             height=raw.shape[0], width=raw.shape[1],
             artifact_cols=right_edge_artifact_width(raw))
    rows.append(q)
    if (i + 1) % 100 == 0:
        print(f"{i + 1}/{len(files)}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(out_dir, "quality.csv"), index=False)
print(df.describe().T.round(3).to_string())
