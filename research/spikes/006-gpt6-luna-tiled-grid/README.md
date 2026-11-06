# Spike 006: GPT-6 Luna (`gpt-6-luna-xhigh`) Parallel 2x2 Tiled Grid + Seam Point-NMS

## Pertanyaan Eksperimen
- **Given:** 9 gambar benchmark (`coins.jpg`, `containers.jpg`, `apples.jpg`, `tennis-ball.jpg`, 3 varian `Heidelbeeren`, dan 2 varian `candy`) yang sebelumnya diuji dengan `gemini-3.8-flash-high` pada Spike 002.
- **When:** Gambar dipotong menjadi grid `2x2` dengan *overlap* 8%, ke-4 tile dikirim secara **paralel** (`ThreadPoolExecutor`) ke model **`gpt-6-luna-xhigh`** (`reasoning_effort: "low"`), lalu hasil titik `[y, x]` tiap tile diproyeksikan kembali ke koordinat global dan digabung menggunakan **Core-Region Filtering + Adaptive Radius Point-NMS**.
- **Then:** Bagaimana akurasi (MAE/MAPE), efisiensi token, kualitas lokalisasi titik, dan latensi *wall-clock* `gpt-6-luna-xhigh` dibandingkan `gemini-3.8-flash-high` pada metode 2x2 Tiled Grid?

---

## Hasil Pengukuran (`results/summary.json`)

| Gambar | Query | Grid | GT | Raw Points (4 Tile) | Setelah Point-NMS | Error | MAPE | Wall Time (s) | Prompt Tok | Comp Tok | Output Image |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `coins.jpg` | `"coins"` | `2x2` | 24 | 25 | **25** | +1 | **4.2%** | 3.68s | 10,432 | 180 | `results/coins__coins__2x2.jpg` |
| `containers.jpg` | `"container"` | `2x2` | 64 | 65 | **64** | 0 | **0.0%** | 13.80s | 10,600 | 1,036 | `results/containers__container__2x2.jpg` |
| `apples.jpg` | `"apples"` | `2x2` | 215 | 353 | **325** | +110 | **51.2%** | 38.60s | 12,028 | 2,904 | `results/apples__apples__2x2.jpg` |
| `tennis-ball.jpg` | `"tennis ball"` | `2x2` | 43 | 42 | **41** | -2 | **4.7%** | 6.83s | 11,672 | 424 | `results/tennis-ball__tennis_ball__2x2.jpg` |
| `Heidelbeeren.jpg` | `"blueberry"` | `2x2` | 113 | 124 | **117** | +4 | **3.5%** | 10.05s | 11,420 | 1,231 | `results/Heidelbeeren__blueberry__2x2.jpg` |
| `Heidelbeeren-bnw.jpg` | `"blueberry"` | `2x2` | 113 | 125 | **116** | +3 | **2.7%** | 13.83s | 11,420 | 1,089 | `results/Heidelbeeren-bnw__blueberry__2x2.jpg` |
| `Heidelbeeren-bnw2.jpg` | `"blueberry"` | `2x2` | 113 | 121 | **116** | +3 | **2.7%** | 16.69s | 11,420 | 2,006 | `results/Heidelbeeren-bnw2__blueberry__2x2.jpg` |
| `candy-ball.jpg` | `"candy"` | `2x2` | 132 | 149 | **144** | +12 | **9.1%** | 9.78s | 11,296 | 903 | `results/candy-ball__candy__2x2.jpg` |
| `candy-bnw.jpg` | `"candy"` | `2x2` | 132 | 119 | **112** | -20 | **15.2%** | 5.61s | 11,296 | 580 | `results/candy-bnw__candy__2x2.jpg` |

---

## Perbandingan Langsung: GPT-6 Luna (`006`) vs Gemini 3.8 Flash (`002`) — 2x2 Tiled Grid

| Gambar | GT | GPT-6 Luna Pred (Err / MAPE) | Gemini 3.8 Flash Pred (Err / MAPE) | GPT-6 Luna Time | Gemini 3.8 Flash Time | Pemenang Akurasi |
|---|---:|---|---|---:|---:|---|
| `coins.jpg` | 24 | 25 (+1 / 4.2%) | **24 (0 / 0.0%)** | **3.68s** | 20.78s | Gemini 3.8 Flash |
| `containers.jpg` | 64 | **64 (0 / 0.0%)** | 65 (+1 / 1.6%) | **13.80s** | 31.53s | GPT-6 Luna (angka), Gemini (lokalisasi titik) |
| `apples.jpg` | 215 | **325 (+110 / 51.2%)** | 336 (+121 / 56.3%) | **38.60s** | 194.73s | GPT-6 Luna |
| `tennis-ball.jpg` | 43 | 41 (-2 / 4.7%) | **42 (-1 / 2.3%)** | **6.83s** | 10.67s | Gemini 3.8 Flash |
| `Heidelbeeren.jpg` | 113 | **117 (+4 / 3.5%)** | 122 (+9 / 8.0%) | **10.05s** | 11.34s | GPT-6 Luna |
| `Heidelbeeren-bnw.jpg` | 113 | **116 (+3 / 2.7%)** | 119 (+6 / 5.3%) | 13.83s | **11.48s** | GPT-6 Luna |
| `Heidelbeeren-bnw2.jpg` | 113 | **116 (+3 / 2.7%)** | 127 (+14 / 12.4%) | 16.69s | **10.98s** | GPT-6 Luna |
| `candy-ball.jpg` | 132 | **144 (+12 / 9.1%)** | 145 (+13 / 9.8%) | **9.78s** | 14.06s | GPT-6 Luna |
| `candy-bnw.jpg` | 132 | 112 (-20 / 15.2%) | **144 (+12 / 9.1%)** | **5.61s** | 12.84s | Gemini 3.8 Flash |
| **Rata-rata (Semua 9 Gambar)** | — | **MAPE: 10.37%** | MAPE: 11.64% | **13.21s** (Total 118.9s) | 35.38s (Total 318.4s) | **GPT-6 Luna (MAPE & Latensi)** |
| **Rata-rata (8 Gambar tanpa `apples`)** | — | **MAPE: 5.26% (MAE: 5.63)** | MAPE: 6.06% (MAE: 7.00) | **10.03s** | 15.46s | **GPT-6 Luna** |

---

## Temuan Utama & Analisis Visual

1. **Akurasi Unggul pada Tumpukan Blueberry (Warna & Monokrom):**
   - Pada ketiga varian `Heidelbeeren` (GT=113), `gpt-6-luna-xhigh` 2x2 sangat konsisten: menghasilkan **117** (RGB, MAPE **3.5%**), **116** (B&W, MAPE **2.7%**), dan **116** (B&W kontras tinggi, MAPE **2.7%**).
   - Sebagai perbandingan, `gemini-3.8-flash-high` 2x2 mengalami *over-counting* lebih tinggi di area perbatasan tile (122, 119, dan 127; MAPE hingga 12.4%). Inspeksi visual pada `Heidelbeeren__blueberry__2x2.jpg` menunjukkan bahwa titik mentah GPT-6 Luna (121–125 titik) jauh lebih bersih di sekitar garis potong (*seam*) dibandingkan Gemini (133–144 titik mentah).

2. **Kecepatan *Wall-Clock* 2.68x Lebih Cepat & Efisiensi Token 7.7x Lebih Hemat:**
   - Total waktu eksekusi seluruh 9 gambar turun dari **318.41 detik** (rata-rata 35.38s/gambar pada Gemini 3.8 Flash) menjadi **118.87 detik** (rata-rata **13.21s/gambar** pada GPT-6 Luna).
   - Total *completion tokens* across 9 gambar hanya **10,353 token** pada `gpt-6-luna-xhigh` dibandingkan **79,731 token** pada `gemini-3.8-flash-high` (yang menghabiskan 64,953 *reasoning tokens*). Pada gambar paling padat (`apples.jpg`), `gpt-6-luna-xhigh` selesai dalam **38.60s** vs **194.73s** pada Gemini.

3. **Kejutan Penting dari Inspeksi Visual (`containers.jpg` vs `candy-bnw.jpg`):**
   - **Angka tepat belum tentu lokalisasi sempurna (`containers.jpg`):** Walaupun `gpt-6-luna-xhigh` menghasilkan angka tepat **64/64 (MAPE 0.0%)** pada `containers.jpg`, inspeksi visual terhadap `results/containers__container__2x2.jpg` memperlihatkan adanya kompensasi dua jenis error: (a) beberapa kontainer yang terlihat dari sudut miring (menampilkan sisi samping panjang dan pintu depan sekaligus di dekat garis potong vertikal) diberi titik ganda pada kedua sisinya, serta terdapat sedikit pergeseran vertikal (*vertical drift*) di tepi atas/bawah; yang kebetulan terkompensasi oleh beberapa kontainer tersembunyi yang terlewat. Untuk lokalisasi koordinat murni pada objek memanjang, Gemini 3.8 Flash masih lebih presisi menempatkan titik tepat di tengah permukaan kontainer.
   - **Penurunan *recall* pada `candy-bnw.jpg` (monokrom kontras rendah):** Saat warna dihilangkan pada `candy-bnw.jpg` (GT=132), `gpt-6-luna-xhigh` menghasilkan 119 titik mentah -> **112** setelah Point-NMS (error -20, MAPE 15.2%), karena melewatkan permen berwarna gelap di area bayangan/pinggir dan beberapa permen di tengah tile. Sebaliknya, pada versi berwarna `candy-ball.jpg`, `gpt-6-luna-xhigh` mencapai MAPE **9.1%** (144 vs 132), sedikit lebih akurat dan 30% lebih cepat daripada Gemini 3.8 Flash (145 vs 132, MAPE 9.8%).

## Cara Menjalankan
```bash
LLM_API_KEY="<api_key_anda>" /root/workspace/count-anything/.venv/bin/python \
  /root/workspace/anycount/research/spikes/006-gpt6-luna-tiled-grid/run_tiled_grid_gpt6.py
```
