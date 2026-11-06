# Spike 003: Side-by-Side Visual & Quantitative Comparison (6 Metode)

Folder ini berisi skrip `compare_all.py` dan 13 gambar montage perbandingan berdampingan (*side-by-side*) untuk ke-6 metode yang diuji:
- **M1:** Count-Anything (SAM3 3.4 GB)
- **M2:** Offline ONNX Model Pack (`2.89 MB`) + LLM Validator + Distilled v2 (`004`)
- **M3:** Gemini 3.8 Flash — Single-Pass (`001`)
- **M4:** Gemini 3.8 Flash — 2x2 Tiled Grid + Point-NMS (`002`)
- **M5:** DeepSeek V4.1 Flash (`deepseek-v4.1-flash-1017`) — Single-Pass (`005`)
- **M6:** DeepSeek V4.1 Flash (`deepseek-v4.1-flash-1017`) — 2x2 Tiled Grid + Point-NMS (`006`)

## Daftar Gambar Montage (`results/`)

1. **`results/montage_cats_and_remotes.jpg`** — Perbandingan M1, M3, dan M5 pada `cats.jpg` (`"cats"` & `"remote control"`).
2. **`results/montage_coins.jpg`** — Perbandingan M3 (`24`), M4 (`24`), M5 (`24`, 3.1s), dan M6 (`25`, 2.8s) pada `coins.jpg`.
3. **`results/montage_containers.jpg`** — Perbandingan M1 (`64`), M3 (`29`), M4 (`65`), M5 (`69`), dan M6 (`61`, 2.5s) pada `containers.jpg`.
4. **`results/montage_apples.jpg`** — Perbandingan M1 (`339`), **M2 Offline ONNX v2 (`290`, 0.21s)**, M3 (`289`), M4 (`336`), dan M6 (`336`, 5.6s) pada `apples.jpg`.
5. **`results/montage_tennis_ball.jpg`** — Perbandingan M1 (`41`), M3 (`42`), M4 (`42`), M5 (`40`, 3.9s), dan M6 (`40`, 3.5s) pada `tennis-ball.jpg`.
6. **`results/montage_heidelbeeren.jpg`** — Perbandingan M1 (`125`), M3 (`122`), M4 (`122`), M5 (`124`), dan M6 (`121`, 3.1s) pada `Heidelbeeren.jpg`.
7. **`results/montage_heidelbeeren_bnw.jpg`** — Perbandingan M1 (`131`), M3 (`112`), M4 (`119`), M5 (`99`), dan M6 (`121`, 3.9s) pada `Heidelbeeren-bnw.jpg`.
8. **`results/montage_heidelbeeren_bnw2.jpg`** — Perbandingan M1 (`137`), M3 (`128`), M4 (`127`), M5 (`105`), dan M6 (`128`, 4.5s) pada `Heidelbeeren-bnw2.jpg`.
9. **`results/montage_candy_ball.jpg`** — Perbandingan M1 (`128`), M3 (`128`), M4 (`145`), M5 (`99`), dan M6 (`142`, 4.7s) pada `candy-ball.jpg`.
10. **`results/montage_candy_bnw.jpg`** — Perbandingan M1 (`123`), M3 (`116`), M4 (`144`), M5 (`132`), dan M6 (`133`, 3.6s) pada `candy-bnw.jpg`.
11. **`results/montage_m2_flywheel_apple_val_1.jpg`** — Metode 2 Flywheel pada `apple_val_1.jpg` (`7` -> Validator `23` -> Distilled v2 **`24`**, 143 ms).
12. **`results/montage_m2_flywheel_apple_val_2.jpg`** — Metode 2 Flywheel pada `apple_val_2.jpg` (`15` -> Validator `41` -> Distilled v2 **`40`**, 139 ms).
13. **`results/montage_m2_flywheel_apples.jpg`** — Metode 2 Flywheel pada `apples.jpg` (`0` -> Validator `289` -> Distilled v2 **`290`**, 212 ms).
