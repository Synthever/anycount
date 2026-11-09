# COCO Pilot Pack — Apple (Colab)

Dua notebook Colab untuk membangun dan menyempurnakan pack detektor `apple`:

1. [`apple_coco_colab.ipynb`](apple_coco_colab.ipynb) — pack dari **COCO 2017** saja:
   unduh anotasi → filter hanya gambar berisi kelas target → unduh gambar itu saja
   (tanpa zip 18 GB) → konversi bbox ke YOLO → latih YOLO11 → mAP + count MAPE →
   ekspor ONNX opset 12 → INT8.
2. [`apple_finetune_colab.ipynb`](apple_finetune_colab.ipynb) — **blend sparse + dense**:
   COCO subset + `apples.jpg` padat (289 titik LLM dari repo) + komposit sintetis
   (40–80 apel per kanvas, label eksak) → latih ulang → evaluasi sparse + dense →
   bandingkan langsung dengan pack `apple-coco` v1 → ekspor INT8 + `best.pt`.

## Hasil notebook 1 (sudah dijalankan user)

YOLO11s, 80 epoch, 1200 COCO train / 75 val, mAP50 0.389, INT8 **9,41 MB**.
Didaftarkan sebagai pack `apple-coco` (query `apple coco`). Pada foto 20 apel user ia
menghitung **19** (M2) versus 35 milik pack distilasi, tetapi hanya 5 pada `apples.jpg`
yang padat — karena itu notebook 2 ada.

## Status verifikasi notebook 2

- Semua cell lolos kompilasi dan JSON valid.
- Dataset + komposit sintetis diuji nyata di VPS (cap 12 COCO + 3 sintetis): label
  dense 289 baris, komposit 13–14 objek dengan label eksak, holdout terpisah dari
  `data.yaml`, `data.yaml` benar.
- Jalur training → eval (mAP + holdout dense + baseline `apple-coco` ONNX) → export
  INT8 → bundle diuji 1 epoch/320 px pada dataset kecil: semua jalan (angka run itu
  tidak bermakna).
- Dry-run menemukan dan memperbaiki dua bug nyata: `Image` dipakai sebelum diimpor, dan
  komposit sintetis sempat ditulis ke direktori yang salah.

## Setelah notebook selesai

Kirim `apple-blend_v1_int8.onnx` + bundle (berisi `best.pt`). Langkah integrasi
(dikerjakan agent):

1. Simpan ONNX ke `research/packs/apple-blend/models/`, daftarkan pack
   (`slug='apple-blend'`, alias mis. `apple blend`, `apel blend`).
2. Uji head-to-head di tiga domain: foto 20 apel (GT 20), `apples.jpg` padat (~289),
   dan beberapa foto COCO-style.
3. Kalau menang di dua domain, pertimbangkan menjadikannya pengganti `apple`.

## Catatan penting

- **Label COCO tidak lengkap untuk tumpukan padat** (`iscrowd` dilewati). Karena itu
  notebook 2 menambah domain padat lewat pseudo-label + komposit sintetis.
- Komposit sintetis memakai potongan apel COCO yang ditempel; label eksak, tetapi
  tampilannya bisa "menempel" — evaluasi dense tetap diukur pada kanvas holdout, dan
  validasi akhir tetap GT manual.
- Lisensi: anotasi COCO CC BY 4.0; gambar dari Flickr dengan lisensi masing-masing.
- Knob notebook 1: `MODEL_WEIGHTS`, `EPOCHS`, `IMG_SIZE`, `MAX_TRAIN_IMAGES`.
  Knob notebook 2 tambahan: `SYNTH_TRAIN`, `SYNTH_TEST`, `SYNTH_MIN_OBJ`, `SYNTH_MAX_OBJ`.
