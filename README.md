# Parmak İzi Doğrulama: Minutiae ve Derin Öznitelikler

BİL462 Biyometrik Tanıma dersi projesi (A1). Aynı veri ve protokol üzerinde minutiae tabanlı
klasik bir eşleştirici ile CNN gömmesini karşılaştırır: DET eğrisi, EER ve sabit FMR'de FNMR.

## Veri

**FVC2002 DB1** (optik sensör, 388×374 px, 500 dpi). Veri lisansı yeniden dağıtıma izin
vermediği için repoda **yoktur**.

1. <http://bias.csr.unibo.it/fvc2002/> adresinden DB1 kümesini indirin.
2. Klasörleri proje köküne koyun:

```
Db1_a/   1_1.tif … 100_8.tif    (test: 100 parmak × 8)
Db1_b/   101_1.tif … 110_8.tif  (eşik/ayar: 10 parmak × 8)
```

## Kurulum

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt   # Linux/macOS
```

## Çalıştırma

Faz 3 (veri ve ön işleme) sonuçlarını yeniden üretmek için:

```bash
python src/phase3_stats.py      # results/phase3/quality.csv
python src/phase3_figures.py    # şekiller, quality_flagged.csv, summary.json
```

Minutiae dalı (SourceAFIS 3.18.1, Java 17+ gerekir):

```bash
bash tools/get_sourceafis.sh                                   # jar'ları tools/lib/ altına indirir
javac -encoding UTF-8 -cp "tools/lib/*" -d build src/minutiae/FvcScores.java
java -cp "build;tools/lib/*" FvcScores Db1_a results/minutiae/scores_sourceafis_Db1_a.csv   # Linux/macOS: ':'
java -cp "build;tools/lib/*" FvcScores Db1_b results/minutiae/scores_sourceafis_Db1_b.csv
python src/evaluate.py results/minutiae/scores_sourceafis_Db1_a.csv --name SourceAFIS     --tune results/minutiae/scores_sourceafis_Db1_b.csv --out results/minutiae
```

Derin dal (ResNet-18; CPU yeterli, PyTorch CPU sürümü için
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`):

```bash
python src/deep/embed_scores.py Db1_a --input gabor     # raw | gabor; --weights ile ince ayarlı model
python src/deep/embed_scores.py Db1_b --input gabor
python src/evaluate.py results/deep/scores_resnet18_gabor_Db1_a.csv --name "ResNet-18 ImageNet gabor"     --tune results/deep/scores_resnet18_gabor_Db1_b.csv --out results/deep
# İnce ayar: FVC2002 DB2–DB4 (DB1 ile ortak kişi yok); CPU'da ~70 dk
python src/deep/finetune.py --epochs 15 --out runs/arcface
python src/deep/embed_scores.py Db1_a --input gabor --weights runs/arcface/best.pt --name resnet18_arcface
python src/deep/embed_scores.py Db1_b --input gabor --weights runs/arcface/best.pt --name resnet18_arcface
python src/evaluate.py results/deep/scores_resnet18_arcface_Db1_a.csv --name "ResNet-18 ArcFace"     --tune results/deep/scores_resnet18_arcface_Db1_b.csv --out results/deep
python src/compare.py --out results/compare     "SourceAFIS=results/minutiae/scores_sourceafis_Db1_a.csv"     "ResNet-18 ArcFace (ince ayar)=results/deep/scores_resnet18_arcface_Db1_a.csv"     "ResNet-18 ImageNet (Gabor)=results/deep/scores_resnet18_gabor_Db1_a.csv"
```

İnce ayar verisi: Springer *Handbook of Fingerprint Recognition* (3. baskı) 4. bölüm ekindeki
zip'ten `FVC2002/Dbs/Db2_*`, `Db3_*`, `Db4_*` klasörlerini proje köküne çıkarın.

Parmak izi görüntüsü içeren şekiller (fig1, fig2, fig5–fig7) FVC lisansı gereği repoda
bulunmaz; `phase3_figures.py` veriyle birlikte çalıştırıldığında yerelde üretilir.

## Yapı

| Yol | İçerik |
|---|---|
| `src/preprocess.py` | Segmentasyon, yön alanı, Gabor iyileştirme, inceltme, minutiae çıkarımı |
| `src/phase3_stats.py` | Görüntü başına kalite ölçüleri |
| `src/phase3_figures.py` | Faz 3 şekilleri, red kuralları ve özet |
| `src/minutiae/FvcScores.java` | FVC protokolüyle SourceAFIS genuine/impostor skorları |
| `src/deep/embed_scores.py` | ResNet-18 gömmesi + kosinüs benzerliği, aynı FVC çiftleri |
| `src/deep/finetune.py` | DB2–DB4 üzerinde ArcFace ince ayarı (doğrulama: Db2–4_b) |
| `src/compare.py` | Sistemlerin ortak DET grafiği ve karşılaştırma tablosu |
| `src/evaluate.py` | EER, FNMR@FMR, parmak bazlı bootstrap GA, DET eğrisi (her iki dal için) |
| `tools/get_sourceafis.sh` | SourceAFIS ve bağımlılıklarını SHA-1 doğrulamalı indirir |
| `results/minutiae/` | Skor CSV'leri, `metrics_sourceafis.json`, DET şekli |
| `results/deep/`, `results/compare/` | CNN skorları/metrikleri, ortak DET ve tablo |
| `results/phase3/` | Grafikler, CSV'ler, `summary.json`, slayt notları |

## Değerlendirme protokolü

- Test: yalnızca DB1-A, FVC protokolü — 2800 genuine (100 × C(8,2)), 4950 impostor (C(100,2)).
- CNN ince ayarı ve eşik seçimi test parmaklarından ayrık verilerle yapılır.
- Kalite kurallarıyla işaretlenen görüntüler testten atılmaz; yalnızca hata analizinde kullanılır.
