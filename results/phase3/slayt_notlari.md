# Faz 3 — Veri ve Ön İşleme Kontrol Noktası (5 dk)

Tüm sayılar `results/phase3/summary.json` ve `quality_flagged.csv` dosyalarından gelir.
Yeniden üretmek için:

    .venv/Scripts/python src/phase3_stats.py
    .venv/Scripts/python src/phase3_figures.py

---

## Slayt 1 — Veri kümesi istatistikleri  (şekil: fig1_dataset_overview.png)
- FVC2002 DB1: optik sensör, 388×374 piksel, 500 dpi, 8-bit gri.
- DB1-A (test): 100 parmak × 8 görüntü = 800 görüntü.
- DB1-B (eşik/ayar): 10 parmak × 8 görüntü = 80 görüntü (101–110). Bu parmaklar A ile ayrık.
- FVC protokolü (DB1-A): 100 × C(8,2) = **2800 genuine**, C(100,2) = **4950 impostor** karşılaştırma.
- Ortalama değerler: ön plan oranı 0,50; sırt periyodu 8,5 px (medyan); minutiae sayısı 32 (20,8 uç + 11,3 çatal).

## Slayt 2 — Ön işleme hattı  (şekil: fig2_pipeline.png)
1. Artefakt temizleme → 2. Blok-varyans segmentasyonu → 3. Normalizasyon →
4. Gradyan tabanlı yön alanı ve tutarlılık (coherence) → 5. Sırt frekansı + yönlü Gabor filtresi →
6. İkili görüntü + inceltme → 7. Crossing-number minutiae ve sahte minutiae temizliği
   (kenara yakın ve 8 px'ten yakın çiftler atılır).
- Görüntü başına yaklaşık 0,25 sn (Python/OpenCV).
- Not: bu hat görselleştirme ve kalite analizi için kuruldu. Asıl eşleştirmede SourceAFIS kullanılacak.

## Slayt 3 — Gözlenen kalite sorunları  (şekiller: fig5_quality_issues.png, fig4_quality_by_impression.png)
- **Sensör artefaktı:** 880 görüntünün 851'inde sağ kenarda 1–3 px koyu dikey çizgi var.
  Segmentasyonu bozduğu için son 4 sütun kırpıldı.
- **Kuru / soluk iz:** sırtlar kesik kesik, yön tutarlılığı düşük. Örnek: 59_7 (tutarlılık = 0,25).
- **Islak / koyu iz:** sırtlar birbirine yapışıyor. Örnek: 42_5.
- **Kayma / kısmi iz:** parmak görüntünün kenarından taşıyor, minutiae sayısı düşük. Örnek: 45_4 (7 minutiae).
- **Küçük temas alanı:** örnek 21_8 (ön plan oranı 0,19).
- **Görüntü numarasına göre sistematik desen:**
  - 4–5. görüntülerde merkez kayması en yüksek.
  - 6. görüntü en koyu ve en geniş temas alanına sahip (fazla basınç / ıslak).
  - 7–8. görüntüler en açık ve en düşük tutarlılığa sahip (kuru).
  - Bu desen veriden ölçüldü. FVC2002 makalesindeki toplama talimatlarıyla karşılaştırılmalı.

## Slayt 4 — Reddedilen örnekler  (şekiller: fig3_quality_hist.png, fig6_rejected_examples.png)
- Kurallar skorlara bakılmadan, önceden sabitlendi:
  - ön plan oranı < 0,25,
  - ortalama tutarlılık < 0,45,
  - minutiae sayısı < 12.
- İşaretlenen görüntüler: **DB1-A'da 41 / 800 (%5,1), DB1-B'de 7 / 80**.
  - Kural bazında: alan 10, tutarlılık 30, minutiae 14 (bir görüntü birden fazla kurala takılabilir).
- Görüntü numarasına göre: 4–5 (kayma) ve 7–8 (kuru) görüntülerinde yoğunlaşıyor.
- Parmak 100'ün 8 görüntüsünden 5'i işaretlendi. Bu parmak tutarlı biçimde kuru.
- **Önemli:** FVC protokolünde test görüntüleri **atılmaz**. İşaretlenen görüntüler karşılaştırmada kalır.
  Eşleştirici bir görüntüyü işleyemezse o karşılaştırmaya en düşük skor verilir (hata olarak sayılır).
  Bu işaretler hata analizinde ("başarısızlıklar kalitesiz görüntülerde mi?") kullanılacak.
- Gabor filtresi kuru izleri büyük ölçüde onarıyor, ama bu görüntülerde sahte minutiae riski artıyor.

## Slayt 5 — Sonraki adımlar
- SourceAFIS ile 7750 FVC karşılaştırması → skor dağılımları, DET eğrisi, EER (hedef: 27 Kasım).
- ResNet-18 gömmeleri: önce ince ayarsız temel sonuç, sonra ayrık parmaklarla ince ayar.
- Parmaklar yeniden örneklenerek bootstrap güven aralığı.
