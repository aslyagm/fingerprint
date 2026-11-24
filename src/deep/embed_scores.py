"""CNN gömmesiyle FVC skorları: ResNet-18 özniteliği + kosinüs benzerliği.

Çiftler SourceAFIS skor dosyasından okunur; böylece iki dal birebir aynı protokolle değerlendirilir.

Giriş varyantları:
    raw    : ham görüntü, ön plan kutusuna kırpılmış, beyazla kare yapılmış
    gabor  : Faz 3 hattının Gabor ile iyileştirilmiş görüntüsü, aynı kırpma

Kullanım:
    python src/deep/embed_scores.py Db1_a --input raw --out results/deep
    python src/deep/embed_scores.py Db1_a --input raw --weights runs/finetuned.pt   # ince ayarlı model
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocess import load, process, segment  # noqa: E402

SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def square_crop(img, mask, fill):
    """Ön plan kutusuna kırp, kısa kenarı `fill` ile doldurarak kare yap, SIZE'a ölçekle."""
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    c = img[y0:y1, x0:x1]
    h, w = c.shape
    s = max(h, w)
    out = np.full((s, s), fill, img.dtype)
    out[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = c
    return cv2.resize(out, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


DPI = {"Db2": 569}  # diğer FVC2002 kümeleri 500 dpi


def prepare_array(path, mode):
    """224×224 float32 gri görüntü (0..1). 500 dpi dışındaki kümeler önce 500 dpi'a ölçeklenir."""
    img = load(str(path))
    dpi = DPI.get(Path(path).parent.name[:3], 500)
    if dpi != 500:
        img = cv2.resize(img, None, fx=500 / dpi, fy=500 / dpi, interpolation=cv2.INTER_AREA)
    if mode == "raw":
        mask, _ = segment(img)
        g = square_crop(img, mask, 255).astype(np.float32) / 255.0
    else:
        r = process(img)
        e = r["enh"].astype(np.float32)
        e = (e - e.min()) / (e.max() - e.min() + 1e-6)
        e[~r["mask"]] = 0.5
        g = square_crop(e, r["mask"], 0.5)
    return g.astype(np.float32)


def to_tensor(g):
    """Gri (H,W) veya (B,1,H,W) 0..1 -> ImageNet normalize edilmiş 3 kanallı tensör."""
    g = torch.as_tensor(g)
    if g.dim() == 2:
        g = g[None]
    g = g.expand(*g.shape[:-3], 3, *g.shape[-2:])
    m = torch.tensor(MEAN).view(3, 1, 1)
    s = torch.tensor(STD).view(3, 1, 1)
    return (g - m) / s


def prepare(path, mode):
    return to_tensor(prepare_array(path, mode))


def build_model(weights_path=None):
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    m.fc = torch.nn.Identity()  # 512-b GAP özniteliği
    if weights_path:
        state = torch.load(weights_path, map_location="cpu")
        m.load_state_dict({k: v for k, v in state.items() if not k.startswith("fc.")}, strict=False)
    return m.eval()


@torch.no_grad()
def embed(model, tensors, batch=32):
    out = []
    for i in range(0, len(tensors), batch):
        out.append(F.normalize(model(torch.stack(tensors[i:i + batch])), dim=1))
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db", help="Db1_a veya Db1_b")
    ap.add_argument("--input", choices=["raw", "gabor"], default="raw")
    ap.add_argument("--weights", help="ince ayarlı ağırlıklar (yoksa ImageNet)")
    ap.add_argument("--name", help="çıktı etiketi (varsayılan: resnet18_<input>)")
    ap.add_argument("--pairs", help="çift listesi (varsayılan: SourceAFIS skor dosyası)")
    ap.add_argument("--out", default="results/deep")
    a = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))

    pairs = pd.read_csv(a.pairs or f"results/minutiae/scores_sourceafis_{a.db}.csv")[["probe", "candidate", "label"]]
    ids = sorted(set(pairs.probe) | set(pairs.candidate), key=lambda s: tuple(map(int, s.split("_"))))

    t0 = time.perf_counter()
    tensors = [prepare(Path(a.db) / f"{i}.tif", a.input) for i in ids]
    t1 = time.perf_counter()
    emb = embed(build_model(a.weights), tensors)
    t2 = time.perf_counter()

    idx = {k: n for n, k in enumerate(ids)}
    p = emb[pairs.probe.map(idx).to_numpy()]
    c = emb[pairs.candidate.map(idx).to_numpy()]
    pairs["score"] = (p * c).sum(1).numpy()

    name = a.name or f"resnet18_{a.input}"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(out / f"scores_{name}_{a.db}.csv", index=False, float_format="%.6f")
    print(f"{name} {a.db}: {len(ids)} görüntü, {len(pairs)} çift | ön işleme {1000*(t1-t0)/len(ids):.0f} ms/görüntü, "
          f"gömme {1000*(t2-t1)/len(ids):.1f} ms/görüntü")


if __name__ == "__main__":
    main()
