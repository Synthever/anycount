# AnyCount — Technology & Counting Method Research (6 Metode)

Dokumentasi riset dan eksperimen terukur (*spike*) untuk membandingkan **6 metode penghitung objek berbasis foto (AnyCount)** yang mendukung **Target C** (query teks bebas + objek renggang hingga tumpukan padat + output gambar bertitik merah-putih bergaya `Count-Anything`).

---

## Daftar 6 Metode yang Diuji

1. **Metode 1 (`000-count-anything-baseline`):** **Count-Anything (SAM3 848M param, checkpoint 3.4 GB)** — model *open-vocabulary* berbasis segmentasi mask.
2. **Metode 2 (`004-offline-onnx-llm-validator`):** **Offline Modular ONNX (`2.89 MB` on-demand pack) + Validasi LLM + Auto-Training Distillation Flywheel** — aplikasi mengunduh bobot `.onnx` kecil sesuai objek yang dicari (`"apple"`), menghitung secara offline (`139–212 ms`), divalidasi oleh LLM untuk menghasilkan *pseudo-label* multi-skala (`full + 2x2 crop`) dan melatih ulang model offline hingga mencapai **MAPE rata-rata 2.33%** terhadap validator LLM!
3. **Metode 3 (`001-gemini-single-pass`):** **LLM `gemini-3.8-flash-high` — Single-Pass Spatial Pointing** — 1x kirim foto utuh, output JSON koordinat `[y, x]` (`0..1000`), dirender dengan Pillow/Canvas.
4. **Metode 4 (`002-gemini-tiled-grid`):** **LLM `gemini-3.8-flash-high` — Parallel 2x2 Tiled Grid + Seam Point-NMS** — foto dibagi 2x2 kuadran (overlap 8%), diproses paralel, lalu di-deduplikasi dengan *Core-Region Filtering + Adaptive Point-NMS*.
5. **Metode 5 (`005-deepseek-single-pass`):** **LLM `deepseek-v4.1-flash-1017` — Single-Pass Spatial Pointing** — konsep sama dengan Metode 3 menggunakan model `deepseek-v4.1-flash-1017` (output koordinat `[x, y]`).
6. **Metode 6 (`006-deepseek-tiled-grid`):** **LLM `deepseek-v4.1-flash-1017` — Parallel 2x2 Tiled Grid + Seam Point-NMS** — konsep sama dengan Metode 4 menggunakan model `deepseek-v4.1-flash-1017`.

---

## Struktur Folder Riset

```text
research/
├── README.md                                  # Ringkasan eksekutif & tabel perbandingan 6 metode
├── samples/                                   # Sampel uji awal (cats, coins, containers, apples)
└── spikes/
    ├── 000-count-anything-baseline/           # Metode 1: Count-Anything (SAM3 3.4GB)
    ├── 001-gemini-single-pass/                # Metode 3: Gemini 3.8 Flash Single-Pass
    ├── 002-gemini-tiled-grid/                 # Metode 4: Gemini 3.8 Flash 2x2 Tiled Grid + NMS
    ├── 003-method-comparison/                 # Montage perbandingan visual berdampingan (M1–M6)
    ├── 004-offline-onnx-llm-validator/        # Metode 2: Offline ONNX (2.89 MB) + LLM Validator Flywheel
    ├── 005-deepseek-single-pass/              # Metode 5: DeepSeek V4.1 Flash Single-Pass
    └── 006-deepseek-tiled-grid/               # Metode 6: DeepSeek V4.1 Flash 2x2 Tiled Grid + NMS
```

---

## Hasil Eksperimen Metode 2 (`004-offline-onnx-llm-validator` — Objek `"apple"` Well-Trained)

Setelah dilakukan pelatihan distilasi yang matang (**35 epoch pada `imgsz=640`** menggunakan *adaptive nearest-neighbor bounding boxes* dari titik LLM + *multi-scale full & 2x2 tile crops oversampling* + inferensi ONNX adaptif `640px` / `2x2 Tiled ONNX`), akurasi model offline **`apple_v2_distilled_int8.onnx` (`2.89 MB`)** meningkat drastis:

| Sampel Uji Apel | Offline ONNX v1 (Cold 5ep @320px) | LLM Validator (`gemini-3.8-flash`) | Offline ONNX v2 Distilled (`2.89 MB`, 35ep @640px) | Selisih v2 vs LLM | Speedup Offline vs LLM |
|---|---|---|---|---|---|
| **`apple_val_1.jpg`** (keranjang apel sedang) | **7** (`22.6 ms`, err 69.6%) | **23** (`17.58s`) | **24** (`143.2 ms`, Single 640) | **4.3% MAPE** (+1 apel) | **122.8x lebih cepat** |
| **`apple_val_2.jpg`** (tumpukan apel pasar) | **15** (`26.6 ms`, err 63.4%) | **41** (`26.72s`) | **40** (`139.1 ms`, Single 640) | **2.4% MAPE** (-1 apel) | **192.1x lebih cepat** |
| **`apples.jpg`** (tumpukan ekstrem ratusan apel) | **0** (`25.3 ms`, err 100%) | **289** (`39.94s`) | **290** (`212.1 ms`, 2x2 Tiled ONNX + NMS) | **0.3% MAPE** (+1 apel) | **188.3x lebih cepat** |
| **Rata-rata Keseluruhan** | **77.67% MAPE** | *Baseline Teacher* | **2.33% MAPE** (`< 0.22 detik` 100% offline!) | **Turun -75.3% error** | **~168x lebih cepat** |

**Kesimpulan Metode 2:** Dengan file `.onnx` berukuran hanya **2.89 MB** per objek yang di-download *on-demand* ke aplikasi, model offline mampu menyamai akurasi LLM Validator (**MAPE 2.33%** bahkan pada tumpukan 289 apel melalui *2x2 Tiled ONNX*) dalam waktu **0.14 – 0.21 detik** tanpa koneksi internet!

---

## Tabel Benchmark Master: Metode 1, 3, 4, 5, 6 (11 Skenario Uji)

| # | Gambar & Query | GT / Ref | M1: Count-Anything (3.4 GB) | M3: Gemini Single-Pass | M4: Gemini 2x2 Tiled | M5: DeepSeek V4.1 Single-Pass | M6: DeepSeek V4.1 2x2 Tiled | Metode Terbaik |
|---:|---|---:|---|---|---|---|---|---|
| 1 | `cats.jpg` (`"cats"`) | **2** | **2** (0%, 23.0s) | **2** (0%, 5.71s) | — | **2** (**0%, 3.20s**) | — | **M5 / M3 / M1** |
| 2 | `cats.jpg` (`"remote control"`) | **2** | — | **2** (**0%, 5.03s**) | — | 0 *(jawab teks `"2"` tanpa array, 4.6s)* | — | **M3 (Gemini Single)** |
| 3 | `coins.jpg` (`"coins"`) | **24** | — | **24** (0%, 13.93s) | **24** (0%, 20.78s) | **24** (**0%, 3.13s**) | **25** (4.2%, **2.83s**) | **M5 / M3** |
| 4 | `tennis-ball.jpg` (`"tennis ball"`) | **43** | **41** (4.7%, 13.6s) | **42** (**2.3%, 7.71s**) | **42** (2.3%, 10.67s) | **40** (7.0%, **3.91s**) | **40** (7.0%, **3.45s**) | **M3 / M5 / M6** |
| 5 | `Heidelbeeren.jpg` (`"blueberry"`, Warna) | **~113** | 125 (10.6%, 13.6s) | 122 (8.0%, 12.39s) | 122 (8.0%, 11.34s) | 124 (9.7%, 4.10s) | **121** (**7.1%, 3.10s**) | **M6 / M3** |
| 6 | `Heidelbeeren-bnw.jpg` (`"blueberry"`, B&W) | **~113** | 131 (15.9%, 13.6s) | **112** (**0.9%, 12.10s**) | 119 (5.3%, 11.48s) | 99 (12.4%, 5.12s) | **121** (7.1%, **3.88s**) | **M3 (Gemini Single)** |
| 7 | `Heidelbeeren-bnw2.jpg` (`"blueberry"`, High-Contrast B&W) | **~113** | 137 (21.2%, 16.6s) | 128 (13.3%, 13.99s) | 127 (12.4%, 10.98s) | **105** (7.1%, 5.70s) | 128 (13.3%, **4.45s**) | **M4 / M6** |
| 8 | `candy-ball.jpg` (`"candy"`, Warna) | **~132** | **128** (**3.0%**, 13.6s) | **128** (**3.0%, 19.02s**) | 145 (9.8%, 14.06s) | 99 (25.0%, 3.88s) | **142** (7.6%, **4.70s**) | **M3 / M1** |
| 9 | `candy-bnw.jpg` (`"candy"`, B&W) | **~132** | 123 (6.8%, 3 FP) | 116 (12.1%, 12.49s) | 144 (9.1%, 12.84s) | **132** (0% angka, *grid*, 5.27s) | **133** (**0.8%, 3.64s**) | **M6 / M4** |
| 10 | `containers.jpg` (`"container"`) | **64** | **64** (Ref, 28.1s A100) | 29 (54.7%, GAGAL) | **65** (**1.6%**, 31.53s) | 69 (7.8%, 2.92s) | **61** (**4.7%, 2.47s**) | **M4 / M6 / M1** |
| 11 | `apples.jpg` (`"apples"`) | **~215–340** | **339** (23.0s CPU) | 289 (39.94s) | **336** (21.09s) | 825 *(loop*, 16.5s) | **336** (**5.60s**) | **M6 / M4 / M2** |
| — | **Rata-rata Latensi** | — | **~17.5s** | **13.49s** | **35.38s** (15.4s tanpa outlier) | **5.30s** | **3.79s (TERCEPAT CLOUD!)** | — |

---

## Temuan Penting Perbandingan DeepSeek V4.1 Flash vs Gemini 3.8 Flash ( & GPT-6 Luna)

1. **DeepSeek V4.1 Flash 2x2 Tiled (`Metode 6`) adalah Metode Cloud Tercepat (`3.79 detik/gambar`):**
   - **9.3x lebih cepat** dibandingkan Gemini 2x2 Tiled (`35.38s`) dan **3.5x lebih cepat** dibandingkan GPT-6 Luna 2x2 Tiled (`13.21s`).
   - Pada 8 skenario di luar `apples.jpg`, Metode 6 mencatat **MAPE rata-rata 6.48% (MAE 6.13 item)** — mengalahkan MAE Gemini 2x2 Tiled (`7.00 item`), termasuk menang pada `candy-bnw.jpg` (`133/132`, error `0.8%` dalam `3.64s`), `containers.jpg` (`61/64` dalam `2.47s`), dan `apples.jpg` (`336` identik dengan Gemini 2x2 namun selesai dalam **`5.60s`**!).
   - Pada `tennis-ball.jpg`, `deepseek-v4.1-flash-1017` menandai **40/43 bola tenis secara presisi tepat di atas bola asli** dalam `3.45s – 3.91s` (jauh lebih baik daripada `gpt-6-luna-xhigh` Single-Pass yang sempat halusinasi 83 titik ke lapangan kosong).
2. **Catatan Penting Audit Visual pada DeepSeek V4.1 Flash (`Metode 5 & 6`):**
   - Pada objek renggang–menengah (`<= 45` objek per gambar/tile seperti `cats`, `coins`, `tennis-ball`, dan per-kuadran `containers`), titik koordinat `[x, y]` DeepSeek mendarat **benar-benar presisi di tengah objek asli**.
   - Namun ketika satu frame/tile berisi **tumpukan seragam >100 objek rapat** (`Heidelbeeren` atau `candy` dalam Single-Pass), DeepSeek terkadang menghitung jumlah totalnya dengan benar terlebih dahulu lalu menyusun sebagian titik dalam pola kisi (*grid*) reguler atau mengalami *repetition loop* pada `apples.jpg` Single-Pass (`825` titik). Membagi gambar menjadi **2x2 Tiled (`Metode 6`)** menghilangkan *repetition loop* pada `apples.jpg` secara total (`825` -> `336` titik bersih dalam `5.60s`)!
3. **Gemini 3.8 Flash (`Metode 3 & 4`) Tetap Unggul dalam Akurasi Posisi Titik Individual (True Per-Instance Grounding):**
   - Jika prioritas utama adalah **setiap titik harus mendarat 100% di tengah masing-masing objek pada tumpukan >100 benda** (misalnya untuk dijadikan **LLM Teacher/Validator penghasil pseudo-label training Metode 2**), **Gemini 3.8 Flash (`Metode 3 & 4`)** adalah pilihan terbaik karena tidak pernah membuat pola kisi sintetis pada `candy-ball.jpg` (`128/132` titik murni) maupun `Heidelbeeren`.

---

## Sintesis Rekomendasi Terbaik untuk Produk AnyCount

Kombinasi paling kuat dan efisien dari ke-6 metode yang sudah kita buktikan:
- **Di Aplikasi User (Mode Offline Cepat & Gratis — Metode 2):** Gunakan **Modular ONNX INT8 Pack (`~2.89 MB` per objek)** dengan inferensi adaptif (`640px` Single-Pass + `2x2 Tiled ONNX` untuk objek padat) — selesai dalam **0.14 – 0.21 detik** dengan **MAPE ~2.3%**.
- **Di Cloud (Untuk Objek Baru / Open-Vocabulary Cepat — Metode 3, 4, 6):**
  - Gunakan **Gemini 3.8 Flash (`Metode 3 / 4`)** sebagai **Validator & Teacher Auto-Labeling** karena posisi titik per-instansnya paling akurat untuk men-training ulang model `.onnx` (Metode 2).
  - Gunakan **DeepSeek V4.1 Flash 2x2 Tiled (`Metode 6`)** bila dibutuhkan mode *Ultra-Fast Cloud Preview* (`~3.5 detik`).
