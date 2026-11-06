# Spike 003: Side-by-Side Visual & Quantitative Comparison

Folder ini berisi skrip `compare_all.py` dan 8 gambar montage perbandingan berdampingan (*side-by-side*) antara:
1. **Count-Anything (SAM3 3.4 GB)**
2. **Gemini 3.8 Flash — Single-Pass (`001`)**
3. **Gemini 3.8 Flash — 2x2 Tiled Grid + Point-NMS (`002`)**

## Daftar Gambar Montage (`results/`)

### Sampel Awal
1. **`results/montage_cats_and_remotes.jpg`** — Count-Anything (`2`) vs Gemini Single-Pass (`2`) vs Open-Vocab `"remote control"` (`2`).
2. **`results/montage_coins.jpg`** — Gemini Single-Pass (`24/24`) vs Gemini 2x2 Tiled + NMS (`24/24`).
3. **`results/montage_containers.jpg`** — Count-Anything (`64`) vs Gemini Single-Pass (`29`) vs Gemini 2x2 Tiled + NMS (`65`).
4. **`results/montage_apples.jpg`** — Count-Anything (`339`) vs Gemini Single-Pass (`289`) vs Gemini 2x2 Tiled + NMS (`336`).

### 4 Gambar Uji Baru (`docs/image/test/`)
5. **`results/montage_tennis_ball.jpg`** — Count-Anything (`41/43`) vs Gemini Single-Pass (`42/43`, 7.7s) vs Gemini 2x2 Tiled (`42/43`, 10.7s).
6. **`results/montage_heidelbeeren.jpg`** — Count-Anything (`125`) vs Gemini Single-Pass (`122`, 12.4s) vs Gemini 2x2 Tiled (`122`, 11.3s).
7. **`results/montage_heidelbeeren_bnw.jpg`** — Count-Anything (`131`, over-count naik saat hitam-putih) vs Gemini Single-Pass (`112`, 12.1s, sangat stabil) vs Gemini 2x2 Tiled (`119`, 11.5s).
8. **`results/montage_candy_ball.jpg`** — Count-Anything (`128`) vs Gemini Single-Pass (`128`, identik dengan Count-Anything tanpa double-dot) vs Gemini 2x2 Tiled (`145`).
