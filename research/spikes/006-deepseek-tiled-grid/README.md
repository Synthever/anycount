# Spike 006: DeepSeek V4.1 Flash (`deepseek-v4.1-flash-1017`) Parallel 2x2 Tiled Grid + Seam Point-NMS

## Pertanyaan Eksperimen
- **Given:** 9 gambar benchmark (`coins.jpg`, `containers.jpg`, `apples.jpg`, `tennis-ball.jpg`, 3 varian `Heidelbeeren`, dan 2 varian `candy`) yang sebelumnya diuji dengan `gemini-3.8-flash-high` (Spike 002) dan `gpt-6-luna-xhigh`.
- **When:** Gambar dipotong menjadi grid `2x2` dengan *overlap* 8%, ke-4 tile dikirim secara **paralel** (`ThreadPoolExecutor`) ke model **`deepseek-v4.1-flash-1017`** menggunakan *natural two-step counting + `[x, y]` coordinate prompt*, lalu hasil titik `[x, y]` tiap tile diproyeksikan kembali ke koordinat global dan digabung menggunakan **Core-Region Filtering + Adaptive Radius Point-NMS**.
- **Then:** Bagaimana akurasi (MAE/MAPE), efisiensi token, kualitas lokalisasi titik, dan latensi *wall-clock* `deepseek-v4.1-flash-1017` dibandingkan `gemini-3.8-flash-high` dan `gpt-6-luna-xhigh` pada metode 2x2 Tiled Grid?

---

## Hasil Pengukuran `deepseek-v4.1-flash-1017` (`results/summary.json`)

| Gambar | Query | Grid | GT | Raw Points (4 Tile) | Setelah Point-NMS | Error | MAPE | Wall Time (s) | Prompt Tok | Comp Tok | Output Image |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `coins.jpg` | `"coins"` | `2x2` | 24 | 25 | **25** | +1 | **4.2%** | 2.83s | 1,148 | 262 | `results/coins__coins__2x2.jpg` |
| `containers.jpg` | `"container"` | `2x2` | 64 | 63 | **61** | -3 | **4.7%** | 2.47s | 1,172 | 580 | `results/containers__container__2x2.jpg` |
| `apples.jpg` | `"apples"` | `2x2` | 215 | 364 | **336** | +121 | **56.3%** | 5.60s | 1,316 | 2,588 | `results/apples__apples__2x2.jpg` |
| `tennis-ball.jpg` | `"tennis ball"` | `2x2` | 43 | 41 | **40** | -3 | **7.0%** | 3.45s | 1,180 | 439 | `results/tennis-ball__tennis_ball__2x2.jpg` |
| `Heidelbeeren.jpg` | `"blueberry"` | `2x2` | 113 | 135 | **121** | +8 | **7.1%** | 3.10s | 1,156 | 1,065 | `results/Heidelbeeren__blueberry__2x2.jpg` |
| `Heidelbeeren-bnw.jpg` | `"blueberry"` | `2x2` | 113 | 136 | **121** | +8 | **7.1%** | 3.88s | 1,156 | 1,049 | `results/Heidelbeeren-bnw__blueberry__2x2.jpg` |
| `Heidelbeeren-bnw2.jpg` | `"blueberry"` | `2x2` | 113 | 142 | **128** | +15 | **13.3%** | 4.45s | 1,156 | 1,112 | `results/Heidelbeeren-bnw2__blueberry__2x2.jpg` |
| `candy-ball.jpg` | `"candy"` | `2x2` | 132 | 162 | **142** | +10 | **7.6%** | 4.70s | 1,160 | 1,225 | `results/candy-ball__candy__2x2.jpg` |
| `candy-bnw.jpg` | `"candy"` | `2x2` | 132 | 152 | **133** | +1 | **0.8%** | 3.64s | 1,160 | 1,137 | `results/candy-bnw__candy__2x2.jpg` |

---

## Perbandingan 3 Model pada 2x2 Tiled Grid: DeepSeek V4.1 Flash vs GPT-6 Luna vs Gemini 3.8 Flash

| Gambar | GT | DeepSeek V4.1 Flash Pred (Err / MAPE) | GPT-6 Luna Pred (Err / MAPE) | Gemini 3.8 Flash Pred (Err / MAPE) | DeepSeek Time | GPT-6 Luna Time | Gemini 3.8 Time | Pemenang Akurasi |
|---|---:|---|---|---|---:|---:|---:|---|
| `coins.jpg` | 24 | 25 (+1 / 4.2%) | 25 (+1 / 4.2%) | **24 (0 / 0.0%)** | **2.83s** | 3.68s | 20.78s | Gemini 3.8 Flash |
| `containers.jpg` | 64 | 61 (-3 / 4.7%) | **64 (0 / 0.0%)** | 65 (+1 / 1.6%) | **2.47s** | 13.80s | 31.53s | GPT-6 Luna (angka), Gemini (lokalisasi) |
| `apples.jpg` | 215 | 336 (+121 / 56.3%) | **325 (+110 / 51.2%)** | 336 (+121 / 56.3%) | **5.60s** | 38.60s | 194.73s | GPT-6 Luna |
| `tennis-ball.jpg` | 43 | 40 (-3 / 7.0%) | 41 (-2 / 4.7%) | **42 (-1 / 2.3%)** | **3.45s** | 6.83s | 10.67s | Gemini 3.8 Flash |
| `Heidelbeeren.jpg` | 113 | 121 (+8 / 7.1%) | **117 (+4 / 3.5%)** | 122 (+9 / 8.0%) | **3.10s** | 10.05s | 11.34s | GPT-6 Luna (DeepSeek #2) |
| `Heidelbeeren-bnw.jpg` | 113 | 121 (+8 / 7.1%) | **116 (+3 / 2.7%)** | 119 (+6 / 5.3%) | **3.88s** | 13.83s | 11.48s | GPT-6 Luna |
| `Heidelbeeren-bnw2.jpg` | 113 | 128 (+15 / 13.3%) | **116 (+3 / 2.7%)** | 127 (+14 / 12.4%) | **4.45s** | 16.69s | 10.98s | GPT-6 Luna |
| `candy-ball.jpg` | 132 | **142 (+10 / 7.6%)** | 144 (+12 / 9.1%) | 145 (+13 / 9.8%) | **4.70s** | 9.78s | 14.06s | **DeepSeek V4.1 Flash** |
| `candy-bnw.jpg` | 132 | **133 (+1 / 0.8%)** | 112 (-20 / 15.2%) | 144 (+12 / 9.1%) | **3.64s** | 5.61s | 12.84s | **DeepSeek V4.1 Flash** |
| **Rata-rata (Semua 9 Gambar)** | — | MAPE: 12.01% (MAE: 18.89) | **MAPE: 10.37%** (MAE: 17.22) | MAPE: 11.64% (MAE: 19.67) | **3.79s** (Total 34.1s) | 13.21s (Total 118.9s) | 35.38s (Total 318.4s) | **GPT-6 Luna (MAPE), DeepSeek (Speed)** |
| **Rata-rata (8 Gambar tanpa `apples`)** | — | MAPE: 6.48% (**MAE: 6.13**) | **MAPE: 5.26% (MAE: 5.63)** | MAPE: 6.06% (MAE: 7.00) | **3.57s** (Total 28.5s) | 10.03s (Total 80.3s) | 15.46s (Total 123.7s) | **GPT-6 Luna (#1), DeepSeek unggul MAE vs Gemini** |

---

## Temuan Utama & Analisis Visual (`deepseek-v4.1-flash-1017`)

1. **Kecepatan *Wall-Clock* Ekstrem (3.5x Lebih Cepat dari GPT-6 Luna, 9.3x Lebih Cepat dari Gemini 3.8 Flash):**
   - Seluruh 9 skenario 2x2 Tiled Grid (36 request tile total) selesai hanya dalam **34.12 detik** (rata-rata **3.79s/gambar**), dibandingkan **118.87 detik** pada `gpt-6-luna-xhigh` (13.21s/gambar) dan **318.41 detik** pada `gemini-3.8-flash-high` (35.38s/gambar).
   - Bahkan pada gambar paling padat (`apples.jpg`, 364 titik mentah), `deepseek-v4.1-flash-1017` menyelesaikan 4 tile paralel hanya dalam **5.60 detik** (vs 38.60s pada GPT-6 Luna dan 194.73s pada Gemini 3.8 Flash).
   - Efisiensi *prompt tokens* sangat tinggi: hanya **10,604 prompt tokens** total untuk 36 tile (~295 token visual+teks per tile), hampir **10x lebih hemat** dibanding GPT-6 Luna (101,584 prompt tokens).

2. **Pemenang Mutlak pada Varian `candy` (`candy-ball.jpg` & `candy-bnw.jpg`):**
   - Pada `candy-ball.jpg` (GT=132), `deepseek-v4.1-flash-1017` mencatat **142** (error +10, MAPE **7.6%**), mengalahkan GPT-6 Luna (144, MAPE 9.1%) dan Gemini 3.8 Flash (145, MAPE 9.8%).
   - Pada `candy-bnw.jpg` (GT=132), saat GPT-6 Luna kehilangan banyak *recall* pada permen gelap di versi monokrom (112, MAPE 15.2%), `deepseek-v4.1-flash-1017` justru sangat tangguh dan menghasilkan **133** (error +1, MAPE **0.8%**). Inspeksi visual pada `results/candy-bnw__candy__2x2.jpg` menunjukkan lokalisasi titik yang presisi di ke-4 kuadran dan *stitching* garis potong vertikal yang bersih (meski terdapat ~5–6 titik ganda di sisi kanan yang saling meniadakan dengan ~5 permen terhalang di bagian dalam).

3. **Karakteristik Prompt & Urutan Koordinat `[x, y]`:**
   - Berbeda dari keluarga Gemini/GPT-6 Luna yang mengeluarkan `[y, x]`, `deepseek-v4.1-flash-1017` secara native mengeluarkan `[x, y]` (horizontal lalu vertikal, ternormalisasi 0..1000).
   - Selain itu, meminta *strict JSON-only* tanpa langkah hitung awal memicu halusinasi deret grid seragam; penggunaan *natural two-step prompt* (`"How many <object> are visible in this image crop? First count them carefully, then output a JSON object..."`) menghasilkan lokalisasi titik objek nyata yang akurat seperti terlihat pada `coins__coins__2x2.jpg` (22 dari 24 koin tepat di titik tengah, hanya +1 duplikat di garis potong vertikal bawah) dan `Heidelbeeren__blueberry__2x2.jpg` (121 vs 113 GT, MAPE 7.1%, lebih baik dari Gemini 3.8 Flash 8.0%).

4. **Keterbatasan pada Perspektif 3D Memanjang (`containers.jpg`):**
   - Pada `containers.jpg` (GT=64), `deepseek-v4.1-flash-1017` menghasilkan **61** (error -3, MAPE **4.7%**) dalam 2.47 detik. Inspeksi visual (`results/containers__container__2x2.jpg`) memperlihatkan fenomena serupa GPT-6 Luna: beberapa kontainer yang terlihat miring (menampilkan sisi panjang sekaligus pintu depan) diberi 2 titik, sedangkan beberapa kontainer di tumpukan atas-tengah (`HAMBURG SÜD` dan kolom tengah-kiri) terlewat.

## Cara Menjalankan
```bash
LLM_API_KEY="<api_key_anda>" /root/workspace/count-anything/.venv/bin/python \
  /root/workspace/anycount/research/spikes/006-deepseek-tiled-grid/run_tiled_grid_deepseek.py
```
