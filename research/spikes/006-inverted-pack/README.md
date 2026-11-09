# Spike 006: Pack yang Dilatih pada Citra Inverted

Pertanyaan lanjutan spike 005: kalau pack dilatih pada citra RGB-negative, apakah ia
memulihkan hitungan pada input inverted — dan apakah itu lebih baik daripada sekadar
memakai pack normal pada citra normal?

## Protokol

- Dataset: `research/packs/tennis-ball` dibangun ulang dengan seed, split, pseudo-box,
  crop 2×2, dan 28 komposit sintetis yang sama; setiap gambar yang ditulis dibalik
  (`ImageOps.invert`) sebelum disimpan → `dataset_inverted/` (239 train / 9 val,
  stem validasi identik dengan dataset normal).
- Latih: `yolo11n.pt`, 35 epoch, `imgsz=640`, `batch=8`, `lr0=0.005`, `mosaic=0.5`,
  `close_mosaic=10`, CPU 6 core — 452 detik. Ekspor ONNX opset 12 + INT8 dinamis
  → `models/tennis_ball_v1_inverted_int8.onnx` (3.03 MB).
- Evaluasi: fungsi worker yang sama dengan M2/M3 (`run_pass`, `run_tiled_pass`), pada
  9 gambar validasi (ref = pseudo-label, rata-rata 3.8 objek/gambar) dan
  `research/samples/tennis-ball.jpg` (ref 43, angka benchmark — belum diverifikasi
  manual). Setiap sel = model × varian input yang sama gambarnya.

## Hasil — matriks 2×2

Sampel padat `tennis-ball.jpg` (count, ref 43):

| Model | Input | M2 single | M3 2×2 |
| --- | --- | --- | --- |
| normal | normal | **41** | 26 |
| normal | inverted | 11 | 0 |
| inverted | inverted | **35** | 18 |
| inverted | normal | 0 | 0 |

Validasi 9 gambar (rata-rata count; pseudo-label mean 3.8):

| Model | Input | M2 single | M3 2×2 |
| --- | --- | --- | --- |
| normal | normal | 0.78 | 0.33 |
| normal | inverted | 0.22 | 0.00 |
| inverted | inverted | 0.22 | 0.11 |
| inverted | normal | 0.11 | 0.00 |

Data mentah: `eval_matrix.json`. Reproduksi:
`python3 train_inverted.py && python3 train_inverted.py --train && python3 eval_matrix.py`
(venv `countthis/pilot/.venv`).

## Observasi

1. **Latih ulang di domain inverted memulihkan sebagian besar kerugian spike 005.**
   Pada sampel padat, M2 naik dari 11 (pack normal, input inverted) menjadi 35 (pack
   inverted, input inverted) — 3,2× lebih baik, mendekati 41 milik pack normal pada
   input normal.
2. **Tapi tidak pernah mengalahkan pasangan normal-normal.** 35 < 41, dan pack inverted
   buta total pada input normal (0 objek). Domain pack harus cocok dengan domain input;
   biaya melatih pack kedua tidak menghasilkan hitungan yang lebih baik.
3. **Validasi terlalu renggang untuk metrik halus.** Rata-rata 3.8 objek/gambar membuat
   MAPE semua sel > 80%; angka sampel padat lebih informatif untuk pertanyaan ini.

## Kesimpulan

Inversi bukan jalur produksi yang menjanjikan: ia hanya berfungsi kalau model juga
dilatih ulang pada warna terbalik, dan hasil terbaiknya tetap kalah dari pipeline normal
pada citra normal. Ini menutup arah (a); preprocessing berikutnya (grayscale/CLAHE) diuji
di spike 007.
