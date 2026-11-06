# Spike 001: Gemini Single-Pass Spatial Pointing

## Pertanyaan Eksperimen
- **Given:** Sebuah foto (`cats.jpg`, `coins.jpg`, `containers.jpg`, `apples.jpg`) dan query teks bebas dari user.
- **When:** Dikirim dalam 1x panggilan (*single-pass*) ke `gemini-3.8-flash-high` dengan instruksi mengembalikan JSON koordinat titik tengah `[y, x]` (`0..1000`), lalu dirender dengan fungsi `_render()` bergaya Count-Anything.
- **Then:** Apakah akurasi jumlah, ketepatan posisi titik, dan latensinya memadai untuk menggantikan Count-Anything?

## Hasil Pengukuran (`results/summary.json`)

| Gambar | Query | GT / Ref | Prediksi | Error | MAPE | Waktu (s) |Output Image |
|---|---|---:|---:|---:|---:|---:|---|
| `cats.jpg` | `"cats"` | 2 | **2** | 0 | **0.0%** | 5.71s | `results/cats__cats.jpg` |
| `cats.jpg` | `"remote control"` | 2 | **2** | 0 | **0.0%** | 5.03s | `results/cats__remote_control.jpg` |
| `coins.jpg` | `"coins"` | 24 | **24** | 0 | **0.0%** | 13.93s | `results/coins__coins.jpg` |
| `containers.jpg` | `"container"` | 64 | **29** | -35 | **54.7%** | 6.04s | `results/containers__container.jpg` |
| `apples.jpg` | `"apples"` | ~215 | **289** | +74 | **34.4%** | 39.94s | `results/apples__apples.jpg` |

## Verdict: PARTIAL (Valid untuk <= 35 objek, Gagal pada objek padat >= 50)

### Apa yang Berhasil
- **Akurasi sempurna (0% error) pada objek renggang hingga menengah (`cats`, `remote control`, `24 coins`):** Seluruh 24 koin pada `coins.jpg` ditandai tepat satu titik per koin tanpa *false positive* di latar belakang.
- **Open-vocabulary sangat fleksibel:** Pada foto `cats.jpg` yang sama, mengganti query dari `"cats"` menjadi `"remote control"` langsung menandai kedua remote TV secara presisi dalam 5 detik.
- **Visual identik dengan Count-Anything:** Dengan memetakan `[y, x]` (`0..1000`) ke piksel gambar dan menggambarnya memakai Pillow, output gambar sama persis dengan `hasil-kucing.jpg`.

### Apa yang Tidak Berhasil
- **Drop akurasi drastis pada objek padat (`containers.jpg`):** Hanya mendeteksi 29 dari 64 kontainer (meleset 35 kontainer), terutama pada tumpukan kontainer tampak belakang/pintu sempit di sisi kiri dan kanan gambar.
- **Latensi tinggi pada ratusan titik (`apples.jpg`):** Men-generate 289 koordinat JSON dalam satu *stream* tunggal memakan waktu ~40 detik.

### Rekomendasi
Gunakan **Single-Pass** sebagai jalur cepat (*fast path*) untuk foto dengan jumlah objek rendah–menengah (`<= 30` objek), dan eskalasikan ke **Spike 002 (Parallel Tiled Grid + Point-NMS)** untuk foto padat.
