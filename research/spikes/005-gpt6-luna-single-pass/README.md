# Spike 005 (Metode 5): GPT-6 Luna XHigh Single-Pass Spatial Pointing

## Pertanyaan Eksperimen
- **Given:** 11 skenario uji (*5 sampel dasar* + *6 foto pengujian baru* termasuk variasi hitam-putih/monokrom) beserta query teks bebas dari user.
- **When:** Dikirim dalam 1x panggilan (*single-pass*) ke model `gpt-6-luna-xhigh` melalui endpoint `/v1/chat/completions` dengan instruksi mengembalikan JSON koordinat titik tengah `[y, x]` (`0..1000`), lalu dirender menggunakan gaya visual Count-Anything (`results/<slug>.jpg`).
- **Then:** Bagaimana akurasi jumlah, ketepatan lokalisasi titik (*spatial grounding*), dan latensinya dibandingkan **Spike 001 (`gemini-3.8-flash-high` Single-Pass)**?

---

## Hasil Pengukuran (`results/summary.json`)

| # | Gambar | Query | GT | Prediksi (`gpt-6-luna-xhigh`) | Error | MAPE (`gpt-6`) | Waktu (`gpt-6`) | Prediksi (`gemini-3.8-flash`) | MAPE (`gemini`) | Waktu (`gemini`) | Output Image |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `cats.jpg` | `"cats"` | 2 | **2** | 0 | **0.0%** | 1.73s | 2 | 0.0% | 5.71s | `results/cats__cats.jpg` |
| 2 | `cats.jpg` | `"remote control"` | 2 | **2** | 0 | **0.0%** | 2.19s | 2 | 0.0% | 5.03s | `results/cats__remote_control.jpg` |
| 3 | `coins.jpg` | `"coins"` | 24 | **24** | 0 | **0.0%** | 3.87s | 24 | 0.0% | 13.93s | `results/coins__coins.jpg` |
| 4 | `containers.jpg` | `"container"` | 64 | **24** | -40 | **62.5%** | 3.18s | 29 | 54.7% | 6.04s | `results/containers__container.jpg` |
| 5 | `apples.jpg` | `"apples"` | 215 | **264** | +49 | **22.8%** | 20.67s | 289 | 34.4% | 39.94s | `results/apples__apples.jpg` |
| 6 | `tennis-ball.jpg` | `"tennis ball"` | 43 | **83** | +40 | **93.0%** | 8.66s | 42 | 2.3% | 7.71s | `results/tennis-ball__tennis_ball.jpg` |
| 7 | `Heidelbeeren.jpg` | `"blueberry"` | 113 | **103** | -10 | **8.8%** | 8.95s | 122 | 8.0% | 12.39s | `results/Heidelbeeren__blueberry.jpg` |
| 8 | `Heidelbeeren-bnw.jpg` | `"blueberry"` | 113 | **133** | +20 | **17.7%** | 11.23s | 112 | 0.9% | 12.10s | `results/Heidelbeeren-bnw__blueberry.jpg` |
| 9 | `Heidelbeeren-bnw2.jpg` | `"blueberry"` | 113 | **92** | -21 | **18.6%** | 8.77s | 128 | 13.3% | 13.99s | `results/Heidelbeeren-bnw2__blueberry.jpg` |
| 10 | `candy-ball.jpg` | `"candy"` | 132 | **102** | -30 | **22.7%** | 8.99s | 128 | 3.0% | 19.02s | `results/candy-ball__candy.jpg` |
| 11 | `candy-bnw.jpg` | `"candy"` | 132 | **101** | -31 | **23.5%** | 20.32s | 116 | 12.1% | 12.49s | `results/candy-bnw__candy.jpg` |

### Ringkasan Metrik Agregat (11 Tugas)

| Metrik | Metode 5: `gpt-6-luna-xhigh` (Single-Pass) | Metode 1: `gemini-3.8-flash-high` (Single-Pass) | Selisih / Perbandingan |
|---|---:|---:|---|
| **Mean Absolute Error (MAE) — Semua 11 Tugas** | **21.91** item | **14.09** item | Gemini **35.7% lebih akurat** secara absolut |
| **Mean MAPE — Semua 11 Tugas** | **24.51%** | **11.70%** | Error persentase `gpt-6-luna-xhigh` **2.1x lebih buruk** |
| **Mean MAPE — 6 Foto Uji Baru (`test/`)** | **30.72%** | **6.60%** | Gemini unggul jauh pada objek menengah–padat (6.6% vs 30.7%) |
| **Akurasi Sempurna (Within ±1 item)** | **3 / 11** tugas (27.3%) | **5 / 11** tugas (45.5%) | Gemini mencapai ±1 pada `tennis-ball` (42/43) & `Heidelbeeren-bnw` (112/113) |
| **Rata-rata Latensi per Gambar** | **8.96 detik** | **13.49 detik** | `gpt-6-luna-xhigh` **33.6% lebih cepat** (0 reasoning tokens) |
| **Total Waktu Sekuensial (11 Gambar)** | **98.56 detik** | **148.35 detik** | Dengan `ThreadPoolExecutor(max_workers=3)` selesai dalam ~38 detik dinding |

---

## Analisis & Temuan Utama

### 1. Keunggulan `gpt-6-luna-xhigh`: Kecepatan Respons & Objek Renggang (`<= 24` objek)
- **Latensi sangat rendah (1.7s – 3.9s pada jumlah objek kecil):** Karena `gpt-6-luna-xhigh` mengembalikan JSON standar (`application/json`) dengan `reasoning_tokens = 0` dan jumlah token keluaran yang ringkas, waktu respons rata-ratanya (8.96s) lebih cepat dibandingkan `gemini-3.8-flash-high` (13.49s).
- **Akurasi hitungan 100% pada objek sedikit hingga 24 item:** Pada `cats.jpg` (`"cats"`: 2/2, `"remote control"`: 2/2) dan `coins.jpg` (`"coins"`: 24/24), jumlah objek yang dikembalikan tepat sasaran (MAPE **0.0%**). Pada `cats__remote_control.jpg`, kedua titik merah mendarat tepat di atas kedua unit remote TV.

### 2. Kelemahan Kritis `gpt-6-luna-xhigh`: *Coordinate Drift* & Halusinasi Titik Spasial
Audit visual pada gambar hasil render (`results/*.jpg`) membuktikan bahwa **angka total yang tampak mendekati GT dapat menyembunyikan kegagalan lokalisasi titik (*spatial grounding*) yang parah**:
- **Halusinasi pada latar belakang & bayangan (`tennis-ball.jpg` — Prediksi 83 vs GT 43, MAPE 93.0%):** Berbeda dengan `gemini-3.8-flash-high` yang menandai 42 dari 43 bola tenis secara presisi (MAPE 2.3%), `gpt-6-luna-xhigh` menghasilkan 83 titik di mana ~55–60 titik melayang di permukaan lapangan biru kosong serta garis bayangan jaring/keranjang, sementara 24 bola tenis asli sama sekali tidak memiliki titik dan 5 bola ditandai 2–4 kali.
- **Pola titik sintetis yang bergeser (*coordinate drift*) pada tumpukan padat (`Heidelbeeren.jpg` — Prediksi 103 vs GT 113, MAPE 8.8%):** Walaupun angka total 103 terlihat dekat dengan 113, inspeksi visual menunjukkan titik-titik merah tidak berpusat pada masing-masing buah blueberry, melainkan tersebar dalam pola yang bergeser: banyak titik jatuh di celah bayangan gelap antar buah, sejumlah buah terkena 2 titik sekaligus, dan banyak buah utuh di latar depan terlewat.
- **Pergeseran skala vertikal pada `coins.jpg` (Prediksi 24 vs GT 24):** Baris 1 dan 2 koin ditandai tepat di tengah, namun baris 3 mulai bergeser ke tepi atas koin, dan pada baris 4 (bawah) titik bergeser ke atas hingga titik koin kanan-bawah menyentuh latar belakang di luar tepi koin.
- **Under-counting konsisten pada `candy` & `containers`:** Pada `candy-ball.jpg` (102 vs 132, -22.7%) dan `candy-bnw.jpg` (101 vs 132, -23.5%), model berhenti di kisaran ~100 titik, sedangkan pada `containers.jpg` hanya mendeteksi 24 dari 64 kontainer (-62.5%).

---

## Verdict: **NO-GO untuk Spatial Pointing Padat (Kalah dari Gemini 3.8 Flash Single-Pass)**

1. **Kemampuan *native spatial pointing* `[y, x]` (`0..1000`) pada keluarga Gemini 3.x jauh lebih presisi dibandingkan `gpt-6-luna-xhigh`.** Gemini dilatih secara khusus untuk tugas *2D point/box grounding*, sehingga titik yang dihasilkan benar-benar menempel di tengah objek pada `tennis-ball` (MAPE 2.3% vs 93.0%), `candy-ball` (MAPE 3.0% vs 22.7%), dan `Heidelbeeren-bnw` (MAPE 0.9% vs 17.7%).
2. **`gpt-6-luna-xhigh` hanya layak dipakai jika jumlah objek sangat sedikit (`<= 10` objek besar)** di mana latensi ~2 detik dibutuhkan, namun **tidak direkomendasikan** untuk menghitung/menandai titik objek menengah–padat (`>= 25` objek) karena tingginya *coordinate drift* dan halusinasi titik di luar objek.
