# Apple COCO Pilot Pack v1 (YOLO11s INT8)

Pack detektor `apple` yang dilatih dari **COCO 2017** (1200 gambar train, 75 val) dengan
notebook `research/packs/coco-pilot/apple_coco_colab.ipynb`. Model: YOLO11s, 80 epoch,
`imgsz=640`, ONNX opset 12 + kuantisasi INT8 **9,41 MB**.

- Slug: `apple-coco` — query `apple coco` / `coco apple` / `apel coco` / `coco`.
- Didaftarkan oleh `seed_coco_apple_pack` di `prototype/src/routes/packs.rs`.
- Pack `apple` yang lama **tidak** diganti: keduanya punya kekuatan berbeda.

## Metadata training (dari `summary.json` Colab)

| Metrik | Nilai |
| --- | --- |
| mAP50 (val COCO) | 0.389 |
| mAP50-95 | 0.228 |
| Count MAPE notebook (30 val, conf 0.25) | 92,0% |
| Train / val images | 1200 / 75 |
| Epochs | 80 |
| ONNX INT8 | 9,41 MB |

MAPE 92% dari notebook dihitung pada val COCO yang renggang (1–5 apel/gambar) dan
menunjukkan model sering undercount di sana; angka itu tidak disimpan sebagai
`val_mape_pct` di DB karena belum bisa diverifikasi ulang di host ini.

## Benchmark hitung (dijalankan di VPS, worker yang sama dengan M2/M3)

| Gambar | GT | Pack lama `apple` M2 / M3 | Pack `apple-coco` M2 / M3 |
| --- | --- | --- | --- |
| foto 20 apel user (736×981) | 20 | 35 / 10 | **19** / 27 |
| `research/samples/apples.jpg` (padat) | ~289 | 320 / 292 | 5 / 14 |

Interpretasi: pack COCO sangat baik di foto apel "biasa" (meleset 1 dari 20) tetapi
runtuh pada tumpukan padat — domain COCO adalah apel pasar/dapur, bukan tumpukan.
Karena itu ia didaftarkan sebagai pack terpisah dengan alias sendiri, bukan menggantikan
pack distilasi yang menang di tumpukan padat.

## Melatih ulang

Jalankan `research/packs/coco-pilot/apple_coco_colab.ipynb` di Colab (T4, Run all),
lalu timpa `models/apple_coco_v1_int8.onnx` dan restart service; seeding akan
memperbarui path/ukuran pack.
