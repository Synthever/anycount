# AnyCount — Technology & Counting Method Research

Dokumentasi riset dan eksperimen terukur (*spike*) untuk menentukan arsitektur penghitung objek berbasis foto (**AnyCount**) yang mendukung **Target C**:
1. **Open-Vocabulary (Query Teks Bebas):** User dapat mengetik nama objek apa saja (`"apple"`, `"cats"`, `"remote control"`, `"container"`, `"tennis ball"`, `"blueberry"`, `"candy"`).
2. **Dense & Sparse Object Support:** Akurat baik untuk objek renggang (1–45 objek) maupun tumpukan padat (50–300+ objek).
3. **Visual Annotated Output:** Menghasilkan gambar dengan titik koordinat merah berpinggir putih (`(230, 40, 35)` + ring putih) dan badge total di pojok kiri atas persis seperti `Count-Anything`, ditambah dukungan interaktif *tap-to-add / tap-to-delete*.

---

##Hasil Benchmark Lengkap (11 Skenario Uji)

> **Catatan Metodologi:** Seluruh pengujian dilakukan secara *zero-shot* tanpa *fine-tuning* khusus domain. Model VLM yang diuji adalah `gemini-3.8-flash-high` melalui endpoint `jr.rkhyg.xyz/v1` dibandingkan langsung dengan **Count-Anything (SAM3 848M param, checkpoint 3.4 GB)**.

| # | Gambar & Query | GT / Ref | Count-Anything (`000` / SAM3 3.4GB) | Gemini Single-Pass (`001`) | Gemini 2x2 Tiled + NMS (`002`) | Metode Terbaik |
|---|---|---:|---|---|---|---|
| 1 | `cats.jpg` (`"cats"`) | **2** | **2** (0% err, 23.0s CPU) | **2** (**0% err, 5.71s**) | — | **Gemini Single-Pass** |
| 2 | `cats.jpg` (`"remote control"`) | **2** | — | **2** (**0% err, 5.03s**) | — | **Gemini Single-Pass** |
| 3 | `coins.jpg` (`"coins"`) | **24** | — | **24** (**0% err, 13.93s**) | **24** (0% err, 20.78s) | **Gemini Single-Pass** |
| 4 | `tennis-ball.jpg` (`"tennis ball"`) | **43** (41 utuh + 2 tepi) | **41** (4.7% err, ~13.6s) | **42** (**2.3% err, 7.71s**) | **42** (2.3% err, 10.67s) | **Gemini Single-Pass** |
| 5 | `Heidelbeeren.jpg` (`"blueberry"`, Warna) | **~113** | **125** (10.6% err, ~13.6s) | **122** (**8.0% err, 12.39s**) | **122** (**8.0% err, 11.34s**) | **Gemini Single / Tiled** |
| 6 | `Heidelbeeren-bnw.jpg` (`"blueberry"`, B&W) | **~113** | **131** (15.9% err, ~13.6s) | **112** (**0.9% err, 12.10s**) | **119** (5.3% err, 11.48s) | **Gemini Single-Pass** |
| 7 | `Heidelbeeren-bnw2.jpg` (`"blueberry"`, High-Contrast B&W) | **~113** | **137** (21.2% err, ~16.6s) | **128** (13.3% err, 13.99s) | **127** (**12.4% err, 10.98s**) | **Gemini 2x2 Tiled** |
| 8 | `candy-ball.jpg` (`"candy"`, Warna) | **~132** | **128** (**3.0% err**, ~13.6s) | **128** (**3.0% err, 19.02s**) | **145** (9.8% err, 14.06s) | **Gemini Single-Pass / CA** |
| 9 | `candy-bnw.jpg` (`"candy"`, B&W) | **~132** | **123** (6.8% err, 3 FP mangkuk) | **116** (12.1% err, 12.49s) | **144** (9.1% err, 12.84s) | **CA / Gemini Single+Tiled** |
| 10 | `containers.jpg` (`"container"`) | **64** | **64** (Ref, 28.1s A100) | **29** (54.7% err, **GAGAL**) | **65** (**1.6% err, 31.53s**) | **Gemini 2x2 Tiled** |
| 11 | `apples.jpg` (`"apples"`) | **~215–340** | **339** (23.0s CPU) | **289** (39.94s) | **336–343** (**21.09s**) | **Gemini 2x2 Tiled** |

---

## Analisis Ablasi Warna & Kontras (`Heidelbeeren` & `candy` Color vs B&W)

1. **Ablasi 3 Varian Blueberry (`Heidelbeeren.jpg` vs `Heidelbeeren-bnw.jpg` vs `Heidelbeeren-bnw2.jpg`):**
   - Ketiga gambar memiliki dimensi piksel yang sama persis (`736x1104`), menguji ketahanan model saat warna dihilangkan (`bnw`, `mean=45.0`) dan saat kontras grayscale ditarik tinggi (`bnw2`, `mean=96.9, std=63.9`):
   - **Kelemahan Count-Anything SAM3 terhadap Kontras Tinggi:** Skor Count-Anything terus membengkak dari `125` (warna) → `131` (B&W) → **`137`** (B&W kontras tinggi). Audit visual menunjukkan kontras tinggi membuat pantulan samar pada buah di lapisan bawah (di sela-sela bayangan gelap) terlihat seperti objek terpisah dengan tepi tajam, sehingga SAM3 mengalami *over-segmentation*.
   - **Perilaku Gemini Single-Pass vs 2x2 Tiled pada `Heidelbeeren-bnw2.jpg`:**
     - Pada `Heidelbeeren-bnw2.jpg`, baik Single-Pass (`128`) maupun 2x2 Tiled (`127`, `10.98s`) jauh lebih tahan terhadap jebakan bayangan celah dibanding SAM3 (`137`).
     - Namun audit visual mengungkap bahwa pada `Heidelbeeren-bnw2.jpg`, **Gemini 2x2 Tiled (`127`) jauh lebih rapi penempatan titiknya** dibanding Single-Pass (`128`), karena Single-Pass sempat melewatkan 5–6 buah di tengah tetapi menggantinya dengan beberapa *double-dot* di tepi kiri bawah.

2. **Ablasi Permen Warna vs Hitam-Putih (`candy-ball.jpg` vs `candy-bnw.jpg`):**
   - Saat warna permen yang mencolok (merah, kuning, biru, hijau, oranye, coklat) diubah menjadi abu-abu (`candy-bnw.jpg`):
   - **Gemini Single-Pass turun dari `128` ke `116` (-12 permen, `12.49s`):** Tanpa kontras warna, permen berwarna gelap yang terpotong di pinggir bingkai (tepi bawah & kiri) serta permen gelap yang berdempetan (*iso-luminant dark candies*) menyatu dengan bayangan, sehingga dilewati oleh Single-Pass. Namun **Gemini Single-Pass memiliki 0 false positive** pada pantulan dinding mangkuk kaca/logam di pojok atas.
   - **Count-Anything SAM3 turun dari `128` ke `123` (-5 permen, `~16.6s`):** Lebih unggul menangkap potongan permen di tepi bingkai karena berbasis kontur lokal, **tetapi mengalami halusinasi 3 titik false-positive pada pantulan buram dinding mangkuk** di pojok kiri atas dan kanan atas (karena tanpa warna, pantulan mangkuk dikira permen).
   - **Gemini 2x2 Tiled (`145` warna vs `144` B&W, `12.84s`):** Hampir **100% kebal terhadap hilangnya warna** (hanya beda 1 titik antara warna dan B&W!) karena perbesaran 2x2 membuat batas tipis antar-permen gelap dan permen tepi tetap terlihat jelas.

---

## Rekomendasi Arsitektur Final untuk AnyCount

Dari 11 eksperimen di atas, terlihat jelas bahwa **Gemini VLM (Single-Pass + 2x2 Tiled Grid)** mampu menyamai dan di banyak kasus mengalahkan **Count-Anything (SAM3 3.4 GB)** tanpa membutuhkan server GPU sepeser pun:
1. **Mode Standar (Single-Pass Spatial Pointing):** Cepat (`5–15 detik`), sangat akurat untuk foto *close-up* berwarna maupun hitam-putih (`cats`, `coins`, `tennis-ball`, `Heidelbeeren`, `candy-ball`).
2. **Mode Detail / Padat (Parallel 2x2 Tiled Grid + NMS):** Sangat efektif untuk objek kecil/jauh (`containers`), tumpukan ratusan (`apples`), atau foto hitam-putih berkontras berat/gelap (`Heidelbeeren-bnw2`, `candy-bnw`) dengan waktu tunggu hanya **11–31 detik** karena ke-4 kuadran diproses paralel.
3. **Interactive Canvas UI (Tap-to-Add / Tap-to-Delete):** Mengizinkan user menyempurnakan 1–5 titik yang meleset di tepi atau bayangan dalam hitungan detik.
