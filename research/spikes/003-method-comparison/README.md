# Spike 003: Side-by-Side Visual & Quantitative Comparison (6 Metode)

Folder ini berisi skrip `compare_all.py` dan 11 gambar montage perbandingan berdampingan (*side-by-side*) untuk ke-6 metode yang diuji:
- **M1:** Count-Anything (SAM3 3.4 GB)
- **M2:** Offline ONNX Model Pack (`2.77 MB`) + LLM Validator + Auto-Retraining Flywheel (`004`)
- **M3:** Gemini 3.8 Flash — Single-Pass (`001`)
- **M4:** Gemini 3.8 Flash — 2x2 Tiled Grid + Point-NMS (`002`)
- **M5:** GPT-6 Luna (`gpt-6-luna-xhigh`) — Single-Pass (`005`)
- **M6:** GPT-6 Luna (`gpt-6-luna-xhigh`) — 2x2 Tiled Grid + Point-NMS (`006`)

## Daftar Gambar Montage (`results/`)

1. **`results/montage_cats_and_remotes.jpg`** — Perbandingan M1, M3, dan M5 pada `cats.jpg` (`"cats"` & `"remote control"`).
2. **`results/montage_coins.jpg`** — Perbandingan M3, M4, M5, dan M6 pada `coins.jpg` (`24` koin).
3. **`results/montage_containers.jpg`** — Perbandingan M1 (`64`), M3 (`29`), M4 (`65`), M5 (`24`), dan M6 (`64`) pada `containers.jpg`.
4. **`results/montage_apples.jpg`** — Perbandingan M1 (`339`), M3 (`289`), M4 (`336`), M5 (`264`), dan M6 (`325`) pada `apples.jpg`.
5. **`results/montage_tennis_ball.jpg`** — Perbandingan M1 (`41`), M3 (`42`), M4 (`42`), M5 (`83` drift), dan M6 (`41`) pada `tennis-ball.jpg`.
6. **`results/montage_heidelbeeren.jpg`** — Perbandingan M1 (`125`), M3 (`122`), M4 (`122`), M5 (`103`), dan M6 (`117`) pada `Heidelbeeren.jpg`.
7. **`results/montage_heidelbeeren_bnw.jpg`** — Perbandingan M1 (`131`), M3 (`112`), M4 (`119`), M5 (`133`), dan M6 (`116`) pada `Heidelbeeren-bnw.jpg`.
8. **`results/montage_heidelbeeren_bnw2.jpg`** — Perbandingan M1 (`137`), M3 (`128`), M4 (`127`), M5 (`92`), dan M6 (`116`) pada `Heidelbeeren-bnw2.jpg`.
9. **`results/montage_candy_ball.jpg`** — Perbandingan M1 (`128`), M3 (`128`), M4 (`145`), M5 (`102`), dan M6 (`144`) pada `candy-ball.jpg`.
10. **`results/montage_candy_bnw.jpg`** — Perbandingan M1 (`123`), M3 (`116`), M4 (`144`), M5 (`101`), dan M6 (`112`) pada `candy-bnw.jpg`.
11. **`results/montage_m2_offline_onnx_flywheel.jpg`** — Bukti visual peningkatan akurasi Metode 2 (`apple_int8.onnx` v1 -> LLM Validator -> Distilled `apple_v2_distilled_int8.onnx`).
