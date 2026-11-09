# COCO Pilot Pack — Apple (Colab)

Notebook satu-file untuk membangun pack detektor `apple` dari **COCO 2017** di Google
Colab: unduh anotasi → filter hanya gambar yang berisi kelas target → unduh gambar itu
saja (tanpa zip 18 GB) → konversi COCO bbox ke YOLO → latih YOLO11 → validasi mAP +
count MAPE → ekspor ONNX opset 12 → kuantisasi INT8 siap dipakai `onnx_worker.py`.

- Notebook: [`apple_coco_colab.ipynb`](apple_coco_colab.ipynb)
- Runtime: T4 GPU, `Runtime → Run all`, ±45–70 menit (80 epoch, 1200 gambar)
- Hasil unduhan otomatis: `apple-coco_v1_int8.onnx` + `apple-coco_v1_bundle.zip`
  (berisi `summary.json`, `results.csv`, plot training)

## Status verifikasi

- Semua cell lolos kompilasi dan JSON notebook valid.
- Dataset path diuji nyata di VPS dengan cap kecil: 20 gambar train + 5 val, 78 bbox,
  label YOLO valid, `data.yaml` benar.
- Jalur training → val → export → INT8 → sanity ONNX diuji 1 epoch pada dataset 20
  gambar (CPU, 320 px): ONNX INT8 2,9 MB, input `(1,3,320,320)`, output `(1,5,2100)`,
  inference onnxruntime berhasil. Angka mAP/MAPE run 1 epoch itu tidak bermakna.

## Setelah notebook selesai

Kirim `apple-coco_v1_int8.onnx` ke agent. Langkah integrasi (dikerjakan agent):

1. Simpan ke `research/packs/apple-coco/models/`.
2. Daftarkan pack: baris `model_packs` (`slug='apple-coco'`,
   aliases `apple, apples, apel`, `imgsz` sesuai notebook, `trained_epochs=80`) atau
   tambahkan seeding di `prototype/src/routes/packs.rs`.
3. Uji hitung terhadap foto GT manual (mis. 20 apel) dan bandingkan dengan pack lama
   di playground (M2/M3).

## Catatan penting

- **Label COCO tidak lengkap untuk tumpukan padat** (`iscrowd` dilewati, dan annotator
  tidak menandai semua instance di kerumunan). Pack ini bagus untuk apel biasa; untuk
  tumpukan rapat tetap perlu pseudo-label/distilasi + GT manual.
- Domain shift: foto COCO adalah apel di pasar/dapur, bukan tumpukan padat.
- Lisensi: anotasi COCO CC BY 4.0; gambar berasal dari Flickr dengan lisensi
  masing-masing — cantumkan atribusi jika dipakai di luar riset.
- Knob penting di cell konfigurasi: `MODEL_WEIGHTS` (`yolo11n/s/m`), `EPOCHS`,
  `IMG_SIZE` (pakai 960 jika objek kecil), `MAX_TRAIN_IMAGES`, dan `USE_DRIVE`.
