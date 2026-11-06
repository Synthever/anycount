# Spike 000: Count-Anything (SAM3 848M / 3.4 GB Checkpoint) Baseline

Folder ini menyimpan skrip `run_ca_batch.py` dan hasil inferensi baseline menggunakan model **Count-Anything (SAM3 848M parameter, checkpoint 3.4 GB)** pada ke-4 gambar uji di `docs/image/test/`.

## Hasil Pengukuran (`results/summary.json`)

| Gambar | Query | Estimasi GT | Count-Anything (`000`) | Error | MAPE | Rata-rata Waktu CPU | Output Image |
|---|---|---:|---:|---:|---:|---:|---|
| `tennis-ball.jpg` | `"tennis ball"` | 43 | **41** | -2 | 4.7% | ~13.6s | `results/tennis-ball__tennis_ball.jpg` |
| `Heidelbeeren.jpg` | `"blueberry"` | 113 | **125** | +12 | 10.6% | ~13.6s | `results/Heidelbeeren__blueberry.jpg` |
| `Heidelbeeren-bnw.jpg` | `"blueberry"` | 110 | **131** | +21 | 19.1% | ~13.6s | `results/Heidelbeeren-bnw__blueberry.jpg` |
| `candy-ball.jpg` | `"candy"` | 132 | **128** | -4 | 3.0% | ~13.6s | `results/candy-ball__candy.jpg` |

## Catatan Audit Visual
- **`tennis-ball.jpg` (41/43):** Melewatkan 2 bola yang terpotong di tepi paling kiri (di bawah net) dan tepi paling kanan (di sebelah klaster kanan bawah).
- **`Heidelbeeren.jpg` (125) vs `Heidelbeeren-bnw.jpg` (131):** Saat informasi warna dihilangkan (hitam-putih), over-count pada Count-Anything meningkat dari 125 menjadi 131 karena cekungan bayangan gelap antar-buah ikut terdeteksi sebagai titik buah tambahan.
- **`candy-ball.jpg` (128/132):** Sangat presisi pada permen yang terlihat utuh maupun separuh; hanya melewatkan ~4 serpihan permen yang terkubur dalam di lapisan bawah.
