"""Demo: iki parmak izi görüntüsünü iki yöntemle karşılaştırır ve kararı yazdırır.

Eşikler test verisinden ayrık DB1-4 B kümesinde FMR %1 hedefiyle seçilmiştir (metrics_*.json).

Kullanım:
    python src/demo.py Db1_a/12_1.tif Db1_a/12_5.tif          # aynı parmak
    python src/demo.py Db1_a/12_1.tif Db1_a/47_1.tif          # farklı parmak
    python src/demo.py A.tif B.tif --show                       # görüntüleri de göster
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "deep"))
from embed_scores import DPI, build_model, prepare  # noqa: E402

TUNE, POINT = "DB1-4 B", "FMR %1"


def threshold(metrics_path):
    m = json.loads((ROOT / metrics_path).read_text(encoding="utf-8"))
    return m["thresholds"][TUNE]["points"][POINT]["threshold"]


def dpi_of(path):
    return DPI.get(Path(path).parent.name[:3], 500)


def sourceafis(a, b):
    sep = ";" if sys.platform == "win32" else ":"
    cp = sep.join([str(ROOT / "build"), str(ROOT / "tools" / "lib" / "*")])
    r = subprocess.run(["java", "-cp", cp, "MatchPair", a, b, str(dpi_of(a)), str(dpi_of(b))],
                       capture_output=True, text=True, check=True)
    return float(r.stdout.strip().splitlines()[-1])


@torch.no_grad()
def cnn(a, b, weights):
    model = build_model(weights)
    e = torch.nn.functional.normalize(model(torch.stack([prepare(a, "gabor"), prepare(b, "gabor")])), dim=1)
    return float((e[0] * e[1]).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--weights", default=str(ROOT / "runs" / "arcface" / "best.pt"))
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    same = Path(a.a).stem.split("_")[0] == Path(a.b).stem.split("_")[0] and Path(a.a).parent == Path(a.b).parent
    print(f"{a.a}  <->  {a.b}   (gerçek: {'aynı parmak' if same else 'farklı parmak'})\n")
    rows = [("SourceAFIS", sourceafis(a.a, a.b), threshold("results/minutiae/metrics_sourceafis.json")),
            ("ResNet-18 ArcFace", cnn(a.a, a.b, a.weights), threshold("results/deep/metrics_resnet-18_arcface.json"))]
    print(f"{'Yöntem':20s} {'Skor':>10s} {'Eşik':>10s}  Karar")
    for name, s, t in rows:
        print(f"{name:20s} {s:10.3f} {t:10.3f}  {'KABUL' if s >= t else 'RED'}")
    print(f"\n(Eşikler: {POINT}, test verisinden ayrık {TUNE} kümesinde seçildi)")

    if a.show:
        import cv2
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(7, 4))
        for x, p in zip(ax, (a.a, a.b)):
            x.imshow(cv2.imread(p, cv2.IMREAD_GRAYSCALE), cmap="gray")
            x.set_title(Path(p).name)
            x.axis("off")
        fig.suptitle(" | ".join(f"{n}: {s:.2f} ({'KABUL' if s >= t else 'RED'})" for n, s, t in rows), fontsize=9)
        plt.show()


if __name__ == "__main__":
    main()
