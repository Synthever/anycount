# AnyCount — Technology & Counting Method Research

Dokumentasi riset dan eksperimen terukur (*spike*) untuk menentukan arsitektur penghitung objek berbasis foto (**AnyCount**) yang mendukung **Target C**:
1. **Open-Vocabulary (Query Teks Bebas):** User dapat mengetik nama objek apa saja (`"apple"`, `"cats"`, `"remote control"`, `"container"`).
2. **Dense & Sparse Object Support:** Akurat baik untuk objek renggang (1–35 objek) maupun tumpukan padat (50–300+ objek).
3. **Visual Annotated Output:** Menghasilkan gambar dengan titik koordinat merah berpinggir putih (`(230, 40, 35)` + ring putih) dan badge total di pojok kiri atas persis seperti `Count-Anything`, ditambah dukungan interaktif *tap-to-add / tap-to-delete*.

---

## Struktur Folder Riset

```text
research/
├── README.md                          # Ringkasan eksekutif, benchmark head-to-head, & rekomendasi arsitektur
├── samples/                           # Dataset sampel standar tanpa anotasi untuk pengujian adil
│   ├── cats.jpg                       # Objek renggang (GT: 2 kucing, 2 remote control)
│   ├── coins.jpg                      # Objek menengah sebagian bersinggungan (GT: 24 koin)
│   ├── containers.jpg                 # Objek padat bertumpuk grid & perspektif (GT/CA: 64 kontainer)
│   ├── containers_ca_points.json      # Ekstraksi 64 titik referensi dari hasil Count-Anything
│   └── apples.jpg                     # Tumpukan sangat padat (~215–340 apel)
└── spikes/
    ├── 001-gemini-single-pass/        # Spike 001: Gemini 3.8 Flash Single-Pass Spatial Pointing
    │   ├── README.md
    │   ├── run_single_pass.py
    │   └── results/                   # Gambar hasil render titik + JSON koordinat + summary.json
    ├── 002-gemini-tiled-grid/         # Spike 002: Gemini 3.8 Flash Parallel Tiled Grid + Point-NMS
    │   ├── README.md
    │   ├── run_tiled_grid.py
    │   └── results/                   # Gambar hasil render grid + JSON koordinat + summary.json
    └── 003-method-comparison/         # Spike 003: Perbandingan Visual Side-by-Side (Montage)
        ├── README.md
        ├── compare_all.py
        └── results/                   # Montage perbandingan Count-Anything vs Gemini Single vs Tiled
```

---

## Temuan Kunci: Mengapa Gemini Bisa Menghasilkan Gambar seperti Count-Anything?

Dari bedah kode `count_anything/inference.py`, model **Count-Anything (SAM3 3.4 GB)** sebenarnya **tidak menghasilkan gambar** secara langsung — model tersebut hanya mengeluarkan daftar koordinat titik `[{"x": ..., "y": ...}]`, lalu fungsi `_render()` sepanjang 25 baris menggunakan library `Pillow (ImageDraw)` untuk menggambar titik merah berpinggir putih di atas foto asli.

Model VLM modern seperti **Gemini (`gemini-3.8-flash-high`)** memiliki kemampuan native **Spatial Pointing** yang dapat mengembalikan koordinat ternormalisasi `[y, x]` dalam skala `0..1000` untuk setiap tengah objek. Dengan menggabungkan output JSON `[y, x]` dari Gemini dan fungsi `_render()` yang sama persis, kita mendapatkan hasil visual yang identik dengan Count-Anything **tanpa perlu menyewa server GPU A100/T4 untuk menahan checkpoint 3.4 GB**.

---

## Hasil Benchmark Head-to-Head

> **Catatan Metodologi:** Angka di bawah adalah hasil pengujian *zero-shot* pada sampel uji standar tanpa *fine-tuning* khusus domain. Waktu respons Gemini diukur melalui gateway `jr.rkhyg.xyz` menggunakan model `gemini-3.8-flash-high`.

| Kasus Uji & Query | Ground Truth / Ref | Count-Anything (SAM3 3.4 GB) | Gemini Single-Pass (`001`) | Gemini 2x2 Tiled + NMS (`002`) |
|---|---:|---|---|---|
| `cats.jpg` (`"cats"`) | **2** | **2** (0% err)<br>23.0s CPU | **2** (0% err)<br>**5.71s** | Tidak perlu tiling (<35 objek) |
| `cats.jpg` (`"remote control"`) | **2** | Belum diuji | **2** (0% err)<br>**5.03s** | Tidak perlu tiling (<35 objek) |
| `coins.jpg` (`"coins"`) | **24** | Belum diuji | **24** (0% err)<br>**13.93s** | **24** (0% err, raw 25 → dedup 24)<br>20.78s |
| `containers.jpg` (`"container"`) | **64** | **64** (Ref)<br>28.1s di Colab A100 | **29** (-54.7% err, **GAGAL**)<br>6.04s | **65** (+1.6% err, raw 71 → dedup 65)<br>**31.53s** |
| `apples.jpg` (`"apples"`) | **~215–340** (tumpukan padat) | **339**<br>23.0s CPU | **289**<br>39.94s | **343** (raw 374 → dedup 343)<br>**21.09s** |

### Analisis Hasil & Audit Visual Titik (Penting!):
1. **Objek Renggang–Menengah (`cats.jpg`, `coins.jpg` — 100% Presisi Murni):**
   - Pada `coins.jpg` (GT: 24 koin dalam grid 4x6), baik **Single-Pass (`13.93s`)** maupun **2x2 Tiled + NMS (`20.78s`)** mencapai **24/24 True Positives (0 false positive, 0 false negative)**. Setiap koin mendapat tepat 1 titik di tengah, dan koin yang tersentuh garis potong tile pada mode `2x2` berhasil di-deduplikasi oleh *Point-NMS* tanpa dihitung ganda.
2. **Objek Padat (`containers.jpg` 64 kontainer — Single-Pass vs Tiled):**
   - **Gemini Single-Pass (`001`) gagal total (29/64, error -54.7%):** Audit visual menunjukkan Single-Pass melewatkan 3 dari 4 kolom tumpukan kontainer di sisi kiri (hanya menandai 7 dari 28 kontainer), melewatkan tumpukan depan *NYK Logistics / Gold*, dan melewatkan 6 kontainer di kolom sempit tengah.
   - **Gemini 2x2 Tiled Grid + Point-NMS (`002`) berhasil memulihkan tumpukan padat (65 vs 64, selisih total +1 / 1.6%):** Seluruh 28 kontainer pada tumpukan 7x4 di sisi kiri berhasil terdeteksi.
   - **Catatan Kejujuran Metrik pada `containers.jpg`:** Walaupun angka total `65` sangat dekat dengan `64`, audit visual per-titik menunjukkan bahwa beberapa kontainer panjang 40-feet yang terbelah garis tengah vertikal (*Hyundai*, *China Shipping*, *NYK Logistics*) atau terlihat dua sisinya sekaligus (sisi panjang + pintu depan) mendapat 2 titik karena jarak kedua titik tersebut (~100 px) lebih lebar dari radius lingkaran *Point-NMS* (`15.2 px`), yang kebetulan saling menutupi dengan 3 kontainer tersembunyi (*recessed*) yang terlewat.
   - **Solusi Arsitektural:** Meminta `[ymin, xmin, ymax, xmax]` (bounding box) alih-alih titik buta `[y, x]` pada mode Tiled memungkinkan *Box-NMS / Seam Stitching* untuk objek memanjang, ditambah **UI Koreksi Manual (Tap-to-Add / Tap-to-Delete)** di aplikasi sebagai pengaman akhir.
3. **Efisiensi Kecepatan pada Objek Ratusan (`apples.jpg`):**
   - Pada tumpukan apel ratusan buah, menjalankan `2x2 Tiled Grid` secara paralel justru **memangkas waktu tunggu hampir 2x lebih cepat** (`21.09s` vs `39.94s` pada Single-Pass) karena 4 stream paralel men-generate ~90 titik per tile jauh lebih cepat daripada 1 stream tunggal yang harus men-generate ~300 titik secara berurutan.

---

## Perbandingan Infrastruktur & Biaya

| Dimensi | Count-Anything (Self-Hosted SAM3) | Gemini Adaptive Hybrid (Single + Tiled) | On-Device Detector + Few-Shot (CountThis Style) |
|---|---|---|---|
| **Ukuran Model / Beban** | Checkpoint 3.4 GB, butuh >7.2 GB RAM / GPU VRAM | **0 MB** di server/HP (via API) | 3–10 MB (ONNX INT8 di HP) |
| **Biaya Server Tetap** | Sewa GPU ~$250–$400/bulan (atau CPU lambat 23s/foto) | **$0 biaya tetap** (pay-per-scan) | **$0** (inferensi di HP) |
| **Kemampuan Query Teks Bebas** | Ya (Open-Vocabulary) | **Ya, Terbaik** (paham konteks/atribut kompleks) | Tidak (terbatas kelas terlatih atau tap contoh) |
| **Kesiapan Produksi MVP** | Berat di-deploy & boros memori | **Siap dalam hitungan hari** | Butuh koleksi dataset & training per kelas |

---

## Rekomendasi Arsitektur untuk AnyCount ("Adaptive Two-Stage Pipeline")

Untuk mencapai efisiensi tertinggi sekaligus memenuhi **Target C**, arsitektur terbaik bukanlah memilih salah satu secara kaku, melainkan **Adaptive Single-to-Tiled VLM Pipeline + Interactive Canvas UI**:

1. **Stage 1 — Fast Single-Pass Scan (Default):**
   - Klien me-resize gambar (sisi panjang maks `1280px`) dan mengirimkannya ke backend bersama `text_query`.
   - Jika jumlah objek yang ditemukan `< 30` dan tidak terdeteksi kepadatan tinggi (ukuran rata-rata jarak antar-titik masih renggang), langsung kembalikan daftar koordinat `[x, y]` ke user (**5–12 detik**).
2. **Stage 2 — Automatic Tiled Grid Escalation (Dense Mode):**
   - Jika user mengaktifkan mode *"Dense / Tumpukan Padat"* atau Stage 1 mendeteksi kepadatan tinggi (`count >= 30` atau objek sangat rapat), sistem otomatis menjalankan **Parallel 2x2 Tiled Grid + Seam Point-NMS** (`002`).
3. **Stage 3 — Interactive Point Overlay & Manual Correction (Di Aplikasi):**
   - Aplikasi merender titik merah-putih secara dinamis di atas kanvas foto (bukan sekadar gambar statis mati).
   - Jika pada tumpukan 64 kontainer model menghitung 65 (meleset 1 titik), user cukup **tap titik yang berlebih untuk menghapus** atau **tap area kosong untuk menambah titik**, lengkap dengan tombol Undo/Redo.
4. **Opsional Tahap Lanjut (Offline / High-Volume):**
   - Untuk kategori yang paling sering dihitung berulang kali oleh user, koordinat hasil Gemini yang sudah dikoreksi user dapat disimpan sebagai dataset otomatis (*auto-labeling*) untuk melatih model lokal ringan (`YOLOv11n` / `RTMDet-tiny` ONNX ~3 MB).
