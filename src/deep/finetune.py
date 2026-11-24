"""ResNet-18'in ArcFace kaybıyla DB1'den ayrık kişiler üzerinde ince ayarı.

FVC2002'de 90 gönüllü 30'ar kişilik üç gruba ayrılmış ve her grup ayrı bir veritabanına atanmıştır;
DB4 sentetiktir (SFinGe). Dolayısıyla DB2–DB4 ile DB1 arasında ortak kişi yoktur.

    eğitim     : Db2_a, Db3_a, Db4_a   (300 parmak × 8)
    doğrulama  : Db2_b, Db3_b, Db4_b   (30 parmak × 8) -> en iyi epoch seçimi (EER)
    test       : Db1_a (eğitimde hiç kullanılmaz)

Kullanım:
    python src/deep/finetune.py --epochs 15 --out runs/arcface
    python src/deep/embed_scores.py Db1_a --input gabor --weights runs/arcface/best.pt --name resnet18_arcface
"""
import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import v2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from embed_scores import prepare_array, to_tensor  # noqa: E402
from evaluate import metrics  # noqa: E402

TRAIN = ["Db2_a", "Db3_a", "Db4_a"]
VAL = ["Db2_b", "Db3_b", "Db4_b"]


def load_set(dbs, mode, cache_dir):
    """Görüntüleri (N,1,224,224) float16 olarak önbellekten yükler/üretir; etiket = küme+parmak."""
    xs, labels, names = [], [], []
    for db in dbs:
        cache = Path(cache_dir) / f"{mode}_{db}.npz"
        if cache.exists():
            d = np.load(cache)
            x, ids = d["x"], list(d["ids"])
        else:
            files = sorted(Path(db).glob("*.tif"), key=lambda p: tuple(map(int, p.stem.split("_"))))
            t0 = time.perf_counter()
            x = np.stack([prepare_array(f, mode) for f in files]).astype(np.float16)
            ids = [f.stem for f in files]
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, x=x, ids=np.array(ids))
            print(f"  önbellek {db}: {len(ids)} görüntü, {time.perf_counter() - t0:.0f} s", flush=True)
        xs.append(x)
        labels += [f"{db}/{i.split('_')[0]}" for i in ids]
        names += [f"{db}/{i}" for i in ids]
    classes = {c: k for k, c in enumerate(sorted(set(labels)))}
    y = np.array([classes[c] for c in labels])
    return torch.from_numpy(np.concatenate(xs)[:, None]), torch.from_numpy(y), names


class ArcFace(nn.Module):
    def __init__(self, dim, n_classes, s=30.0, m=0.3):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_classes, dim) * 0.01)
        self.s, self.m = s, m

    def forward(self, emb, y):
        cos = F.linear(F.normalize(emb), F.normalize(self.W)).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cos.gather(1, y[:, None]))
        target = torch.cos(theta + self.m)
        logits = cos.scatter(1, y[:, None], target)
        return F.cross_entropy(self.s * logits, y)


def backbone():
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Identity()
    return m


@torch.no_grad()
def embed(model, x, batch=64):
    model.eval()
    return torch.cat([F.normalize(model(to_tensor(x[i:i + batch].float())), dim=1)
                      for i in range(0, len(x), batch)])


def fvc_pairs(names):
    """Her küme içinde FVC protokolü: tüm genuine çiftler + 1. izlenimler arası impostor."""
    by_finger = {}
    for k, n in enumerate(names):
        db, img = n.split("/")
        f, imp = img.split("_")
        by_finger.setdefault((db, f), {})[int(imp)] = k
    gen, imp = [], []
    for v in by_finger.values():
        ks = [v[i] for i in sorted(v)]
        gen += [(a, b) for i, a in enumerate(ks) for b in ks[i + 1:]]
    keys = sorted(by_finger)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if a[0] == b[0]:
                imp.append((by_finger[a][min(by_finger[a])], by_finger[b][min(by_finger[b])]))
    return np.array(gen), np.array(imp)


def val_eer(model, x, gen_pairs, imp_pairs):
    e = embed(model, x)
    g = (e[gen_pairs[:, 0]] * e[gen_pairs[:, 1]]).sum(1).numpy()
    i = (e[imp_pairs[:, 0]] * e[imp_pairs[:, 1]]).sum(1).numpy()
    return metrics(g, i)["eer"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", choices=["raw", "gabor"], default="gabor")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default="runs/arcface")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="hız testi: epoch başına en fazla bu kadar batch")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    print("Veri hazırlanıyor...", flush=True)
    xtr, ytr, _ = load_set(TRAIN, a.input, a.cache)
    xva, _, nva = load_set(VAL, a.input, a.cache)
    gp, ip = fvc_pairs(nva)
    n_cls = int(ytr.max()) + 1
    print(f"eğitim {len(xtr)} görüntü / {n_cls} parmak | doğrulama {len(xva)} görüntü, "
          f"{len(gp)} genuine, {len(ip)} impostor", flush=True)

    aug = v2.Compose([
        v2.RandomAffine(degrees=25, translate=(0.1, 0.1), scale=(0.9, 1.1), fill=0.5),
        v2.RandomErasing(p=0.3, scale=(0.02, 0.15), value=0.5),
    ])
    model = backbone()
    head = ArcFace(512, n_cls)
    opt = torch.optim.AdamW([{"params": model.parameters(), "lr": a.lr},
                             {"params": head.parameters(), "lr": a.lr * 10}], weight_decay=1e-4)
    steps = a.epochs * math.ceil(len(xtr) / a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, a.lr * 10], total_steps=steps, pct_start=0.1)

    best = val_eer(model, xva, gp, ip)
    print(f"epoch 0 (ImageNet): doğrulama EER %{100 * best:.2f}", flush=True)
    torch.save(model.state_dict(), out / "best.pt")
    log = open(out / "log.csv", "w", newline="")
    w = csv.writer(log)
    w.writerow(["epoch", "loss", "val_eer", "seconds"])
    w.writerow([0, "", best, 0])

    for ep in range(1, a.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        perm = torch.randperm(len(xtr))
        losses = []
        for b, i in enumerate(range(0, len(xtr), a.batch)):
            if a.limit and b >= a.limit:
                break
            idx = perm[i:i + a.batch]
            xb = torch.stack([aug(x) for x in xtr[idx].float()])
            loss = head(model(to_tensor(xb)), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        if a.limit:
            dt = time.perf_counter() - t0
            print(f"hız testi: {len(losses)} batch {dt:.1f} s -> tam epoch ~"
                  f"{dt / len(losses) * math.ceil(len(xtr) / a.batch) / 60:.1f} dk", flush=True)
            return
        eer = val_eer(model, xva, gp, ip)
        dt = time.perf_counter() - t0
        w.writerow([ep, np.mean(losses), eer, round(dt)])
        log.flush()
        mark = ""
        if eer < best:
            best = eer
            torch.save(model.state_dict(), out / "best.pt")
            mark = "  * en iyi"
        print(f"epoch {ep}/{a.epochs}: kayıp {np.mean(losses):.3f}, doğrulama EER %{100 * eer:.2f}, "
              f"{dt / 60:.1f} dk{mark}", flush=True)
    print(f"Bitti. En iyi doğrulama EER %{100 * best:.2f} -> {out / 'best.pt'}")


if __name__ == "__main__":
    main()
