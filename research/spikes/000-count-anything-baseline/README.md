# Spike 000: Count-Anything (SAM3 848M / 3.4 GB Checkpoint) Baseline

Folder ini menyimpan skrip `run_ca_batch.py` dan hasil inferensi baseline menggunakan model **Count-Anything (SAM3 848M parameter, checkpoint 3.4 GB)** pada ke-6 gambar uji di `docs/image/test/`.

## Hasil Pengukuran (`results/summary.json`)

| Gambar | Query | Estimasi GT | Count-Anything (`000`) | Error | MAPE | Rata-rata Waktu CPU | Output Image |
|---|---|---:|---:|---:|---:|---:|---|
| `tennis-ball.jpg` | `"tennis ball"` | 43 | **41** | -2 | 4.7% | ~13.6s | `results/tennis-ball__tennis_ball.jpg` |
| `Heidelbeeren.jpg` | `"blueberry"` | 113 | **125** | +12 | 10.6% | ~13.6s | `results/Heidelbeeren__blueberry.jpg` |
| `Heidelbeeren-bnw.jpg` | `"blueberry"` | 113 | **131** | +18 | 15.9% | ~13.6s | `results/Heidelbeeren-bnw__blueberry.jpg` |
| `Heidelbeeren-bnw2.jpg` | `"blueberry"` | 113 | **137** | +24 | 21.2% | ~16.6s | `results/Heidelbeeren-bnw2__blueberry.jpg` |
| `candy-ball.jpg` | `"candy"` | 132 | **128** | -4 | 3.0% | ~13.6s | `results/candy-ball__candy.jpg` |
| `candy-bnw.jpg` | `"candy"` | 132 | **123** | -9 | 6.8% | ~16.6s | `results/candy-bnw__candy.jpg` |

## Catatan Audit Visual
- **Sensitivitas Kontras & Warna pada Blueberry (`Heidelbeeren` 3 varian):**
  - Ketiga file adalah foto yang sama persis (`736x1104`): warna asli (`125`), grayscale normal (`131`), dan grayscale kontras tinggi (`137`).
  - Saat warna dihapus dan kontras dinaikkan (`Heidelbeeren-bnw2.jpg`), over-count Count-Anything melonjak ke **137** karena pantulan cahaya samar di celah bayangan dalam (*interstitial crevices*) ikut tersegmentasi sebagai buah terpisah.
- **Halusinasi Pantulan Wadah pada `candy-bnw.jpg` (`123`):**
  - Saat warna permen dihapus (`candy-bnw.jpg`), Count-Anything menempatkan **3 titik false-positive di dinding mangkuk yang buram** (pojok kiri atas dan kanan atas), karena tanpa informasi warna SAM3 mengira pantulan lengkung mangkuk sebagai permen.
