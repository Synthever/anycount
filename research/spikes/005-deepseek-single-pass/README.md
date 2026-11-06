# Spike 005 (Metode 5): DeepSeek V4.1 Flash (`deepseek-v4.1-flash-1017`) Single-Pass Spatial Pointing

## Pertanyaan Eksperimen
- **Given:** 11 skenario uji (*5 sampel dasar* + *6 foto pengujian baru* termasuk variasi hitam-putih/monokrom) beserta query teks bebas dari user.
- **When:** Dikirim dalam 1x panggilan (*single-pass*) ke model `deepseek-v4.1-flash-1017` melalui endpoint `/v1/chat/completions` menggunakan prompt alami dua tahap (*count-then-locate*) yang meminta koordinat titik tengah `[x, y]` ternormalisasi (`0..1000`, di mana `x` horizontal kiri-ke-kanan dan `y` vertikal atas-ke-bawah), lalu dirender menggunakan gaya visual Count-Anything (`results/<slug>.jpg`).
- **Then:** Bagaimana akurasi jumlah, ketepatan lokalisasi titik (*spatial grounding*), dan latensinya dibandingkan **Spike 001 (`gemini-3.8-flash-high` Single-Pass)** serta eksperimen awal **`gpt-6-luna-xhigh` Single-Pass**?

---

## Karakteristik Teknis `deepseek-v4.1-flash-1017` yang Terverifikasi
1. **Format Respons HTTP (`JSON + data: [DONE]`):** Endpoint mengembalikan objek JSON standar yang langsung diikuti string `data: [DONE]` pada baris yang sama, sehingga perlu di-parse menggunakan `json.JSONDecoder().raw_decode(raw_body.lstrip())`.
2. **Urutan Koordinat `[x, y]` (Bukan `[y, x]`):** Berbeda dengan keluarga Gemini yang menggunakan `[y, x]`, `deepseek-v4.1-flash-1017` secara native mengeluarkan `[x, y]` (`nx` horizontal `0..1000`, `ny` vertikal `0..1000`).
3. **Sensitivitas Prompt (*Count-then-Locate*):** Jika diminta *"Return ONLY JSON"*, model langsung menghalusinasikan kisi seragam (*uniform grid*). Dengan prompt alami dua tahap (*"How many <query> are visible in this image? First count them carefully, then output a JSON object..."*), model menghitung dan menempatkan titik secara akurat pada objek renggang hingga menengah (`<= 43` objek).

---

## Hasil Pengukuran (`results/summary.json`)

| # | Gambar | Query | GT | Prediksi (`deepseek-v4.1-flash`) | Error | MAPE (`deepseek`) | Waktu (`deepseek`) | Prediksi (`gemini-3.8-flash`) | MAPE (`gemini`) | Waktu (`gemini`) | Prediksi (`gpt-6-luna-xhigh`) | MAPE (`gpt-6`) | Output Image |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `cats.jpg` | `"cats"` | 2 | **2** | 0 | **0.0%** | 3.20s | 2 | 0.0% | 5.71s | 2 | 0.0% | `results/cats__cats.jpg` |
| 2 | `cats.jpg` | `"remote control"` | 2 | **0** *(teks `"2"` tanpa titik)* | -2 | **100.0%** | 4.60s | 2 | 0.0% | 5.03s | 2 | 0.0% | `results/cats__remote_control.jpg` |
| 3 | `coins.jpg` | `"coins"` | 24 | **24** | 0 | **0.0%** | 3.13s | 24 | 0.0% | 13.93s | 24 | 0.0% | `results/coins__coins.jpg` |
| 4 | `containers.jpg` | `"container"` | 64 | **69** | +5 | **7.8%** | 2.92s | 29 | 54.7% | 6.04s | 24 | 62.5% | `results/containers__container.jpg` |
| 5 | `apples.jpg` | `"apples"` | 215 | **825** *(loop)* | +610 | **283.7%** | 16.50s | 289 | 34.4% | 39.94s | 264 | 22.8% | `results/apples__apples.jpg` |
| 6 | `tennis-ball.jpg` | `"tennis ball"` | 43 | **40** | -3 | **7.0%** | 3.91s | 42 | 2.3% | 7.71s | 83 | 93.0% | `results/tennis-ball__tennis_ball.jpg` |
| 7 | `Heidelbeeren.jpg` | `"blueberry"` | 113 | **124** | +11 | **9.7%** | 4.10s | 122 | 8.0% | 12.39s | 103 | 8.8% | `results/Heidelbeeren__blueberry.jpg` |
| 8 | `Heidelbeeren-bnw.jpg` | `"blueberry"` | 113 | **99** | -14 | **12.4%** | 5.12s | 112 | 0.9% | 12.10s | 133 | 17.7% | `results/Heidelbeeren-bnw__blueberry.jpg` |
| 9 | `Heidelbeeren-bnw2.jpg` | `"blueberry"` | 113 | **105** | -8 | **7.1%** | 5.70s | 128 | 13.3% | 13.99s | 92 | 18.6% | `results/Heidelbeeren-bnw2__blueberry.jpg` |
| 10 | `candy-ball.jpg` | `"candy"` | 132 | **99** | -33 | **25.0%** | 3.88s | 128 | 3.0% | 19.02s | 102 | 22.7% | `results/candy-ball__candy.jpg` |
| 11 | `candy-bnw.jpg` | `"candy"` | 132 | **132** | 0 | **0.0%** | 5.27s | 116 | 12.1% | 12.49s | 101 | 23.5% | `results/candy-bnw__candy.jpg` |

### Ringkasan Metrik Agregat (11 Tugas)

| Metrik | `deepseek-v4.1-flash-1017` (Single-Pass) | `gemini-3.8-flash-high` (Spike 001) | `gpt-6-luna-xhigh` (Run Awal) | Catatan Perbandingan |
|---|---:|---:|---:|---|
| **Mean Absolute Error (MAE) — Semua 11 Tugas** | **62.36** item | **14.09** item | **21.91** item | MAE `deepseek` terdistorsi oleh *repetition loop* pada `apples.jpg` (+610) |
| **Mean MAPE — Semua 11 Tugas** | **41.15%** | **11.70%** | **24.51%** | Tanpa outlier `apples.jpg` (10 tugas), MAPE `deepseek` adalah **16.90%** |
| **Mean MAE — 6 Foto Uji Baru (`test/`)** | **11.50** item | **7.17** item | **25.50** item | `deepseek` jauh lebih baik dari `gpt-6-luna-xhigh` pada 6 foto uji baru |
| **Mean MAPE — 6 Foto Uji Baru (`test/`)** | **10.20%** | **6.60%** | **30.72%** | `deepseek` mendekati Gemini (10.20% vs 6.60%) dan 3x lebih baik dari `gpt-6` |
| **Akurasi Sempurna (Within ±1 item)** | **3 / 11** tugas (27.3%) | **5 / 11** tugas (45.5%) | **3 / 11** tugas (27.3%) | Tepat 100% pada `cats` (2/2), `coins` (24/24), dan angka `candy-bnw` (132/132) |
| **Rata-rata Latensi per Gambar** | **5.30 detik** | **13.49 detik** | **8.96 detik** | `deepseek-v4.1-flash-1017` adalah yang **paling cepat** (**2.5x lebih cepat** dari Gemini) |
| **Total Waktu Sekuensial (11 Gambar)** | **58.33 detik** | **148.35 detik** | **98.56 detik** | Mayoritas gambar selesai dalam **2.9s – 5.7s** (0 reasoning tokens) |

---

## Audit Visual Lokalisasi Titik (*Spatial Grounding*)

Pemeriksaan visual langsung pada gambar hasil render (`results/*.jpg`) menunjukkan pemisahan perilaku yang sangat jelas berdasarkan kepadatan objek:

### 1. Akurat & Presisi pada Objek Renggang hingga Menengah (`<= 43` Objek)
- **`cats__cats.jpg` (Prediksi 2 vs GT 2, MAPE 0.0%, 3.20s):** Kedua titik merah mendarat tepat di tubuh masing-masing kucing (satu di bagian perut/rusuk kucing kiri `[260, 431]`, satu di bagian dada/bahu atas kucing kanan `[725, 435]`).
- **`coins__coins.jpg` (Prediksi 24 vs GT 24, MAPE 0.0%, 3.13s):** Ke-24 koin ditandai satu-per-satu secara akurat tanpa ada koin yang terlewat, tanpa duplikasi, dan tanpa titik yang keluar ke latar belakang (jauh lebih stabil dibandingkan `gpt-6-luna-xhigh` yang mengalami pergeseran skala vertikal pada baris bawah).
- **`tennis-ball__tennis_ball.jpg` (Prediksi 40 vs GT 43, MAPE 7.0%, 3.91s):** Ke-40 titik merah mendarat **langsung di atas bola tenis asli** tanpa ada satu pun titik yang melayang di lapangan biru kosong atau membentuk kisi sintetis (memperbaiki kegagalan fatal `gpt-6-luna-xhigh` yang menghasilkan 83 titik di permukaan lapangan kosong). Hanya 3 bola di kerumunan rapat kanan-bawah/tepi gambar yang terlewat.

### 2. Perilaku Semi-Grid & *Coordinate Repetition Loop* pada Tumpukan Padat (`>= 64` Objek)
- **`containers__container.jpg` (Prediksi 69 vs GT 64, MAPE 7.8%, 2.92s):** Seluruh 69 titik berada di dalam area fisik tumpukan kontainer (tidak ada yang bocor ke langit atau aspal), namun pada kontainer panjang 40-kaki di latar depan (`HYUNDAI`, `CHINA SHIPPING`), model menempatkan 4 titik berjarak teratur sepanjang dinding samping kontainer sehingga membentuk kolom vertikal.
- **`Heidelbeeren*.jpg` (99–124 titik) & `candy*.jpg` (99 & 132 titik):** Meskipun total angka yang diprediksi sangat dekat dengan Ground Truth (bahkan tepat **132/132** pada `candy-bnw.jpg` dan MAPE **7.1%–12.4%** pada ketiga foto `Heidelbeeren`), audit visual membuktikan bahwa pada tumpukan padat ~100+ item, `deepseek-v4.1-flash-1017` menyusun koordinat dalam **pola kisi baris-dan-kolom sintetis** (*synthetic grid*). Karena objek memenuhi hampir 100% kanvas gambar, titik-titik kisi tersebut jatuh di atas tumpukan buah/permen, tetapi tidak berpusat satu-per-satu pada tiap butir (beberapa jatuh di celah bayangan dan beberapa butir terkena 2 titik).
- **`apples__apples.jpg` (Prediksi 825 vs GT 215, MAPE 283.7%, 16.50s):** Pada gambar dengan >200 apel, model terjebak dalam *coordinate repetition loop* hingga mencapai 4.980 token keluaran (menghasilkan 825 titik yang terdiri dari 406 koordinat unik di atas kisi 21x21 dengan ratusan titik bertumpuk di tepi gambar).
- **`cats__remote_control.jpg` (Prediksi 0 titik, teks `"2"`):** Dengan bentuk tunggal `"How many remote control are visible..."`, model hanya menjawab string `"2"` (2 token) tanpa melanjutkan ke blok JSON koordinat (sedangkan jika diprompt dengan bentuk jamak `"remote controls"`, model mengeluarkan `"2\n[[166, 188], [544, 267]]"`).

---

## Verdict: **Cukup Baik untuk Objek Renggang–Menengah (`<= 45` item) & Estimasi Cepat, Namun Gemini 3.8 Flash Tetap Unggul untuk Point Grounding Padat**

1. **Peningkatan Signifikan dibanding `gpt-6-luna-xhigh` pada 6 Foto Uji Baru (`test/`):** Mean MAPE pada 6 foto uji turun tajam dari **30.72%** (`gpt-6-luna-xhigh`) menjadi **10.20%** (`deepseek-v4.1-flash-1017`), dengan kecepatan rata-rata **5.30 detik/gambar** (2.5x lebih cepat dari `gemini-3.8-flash-high`). Lokalisasi titik pada `tennis-ball.jpg` (40/43) dan `coins.jpg` (24/24) benar-benar menempel pada objek.
2. **Keterbatasan pada Tumpukan Padat (`>= 90` objek):** Untuk aplikasi yang hanya membutuhkan **estimasi angka total** pada tumpukan ~100–135 item (`Heidelbeeren`, `candy`), angka yang dikeluarkan `deepseek-v4.1-flash-1017` memiliki error rendah (MAPE 7%–12%), tetapi koordinat titiknya berubah menjadi kisi sintetis dan berisiko mengalami *repetition loop* pada >200 objek (`apples.jpg`). Untuk *true per-instance point grounding* pada objek padat, **`gemini-3.8-flash-high` (Spike 001/002)** tetap menjadi pilihan utama.
