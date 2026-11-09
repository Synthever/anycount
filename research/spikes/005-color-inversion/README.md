# Spike 005: Inversi Warna (RGB Negative) sebagai Preprocessing Counting

Pertanyaan: apakah membalik warna gambar (`R,G,B → 255−nilai`) sebelum counting membuat
objek lebih mudah dideteksi oleh metode AnyCount?

Fitur yang diuji: flag `invert` pada `POST /api/runs` (lihat `prototype/README.md`).
Derivatif dibuat dari `work_path` (JPEG ≤1280px q90) sehingga satu-satunya variabel yang
berubah adalah warna, bukan resolusi atau kompresi. Semua metode dan renderer membaca
derivatif yang sama; `GET /api/runs/{id}/raw` menyajikan citra yang benar-benar dihitung.

Data mentah: `ab_runs.json` (dihasilkan `run_ab.py`). Service: `anycount-prototype`
release build, 2026-11-09, `127.0.0.1:2030`.

## Hasil 1 — Metode offline (deterministik, 1 run per sel)

Pack yang tersedia hanya apple, cat, tennis-ball; kelas lain cold-start di M2/M3.

| Gambar | Metode | Baseline | Inverted | Selisih | Waktu baseline → inverted |
| --- | --- | --- | --- | --- | --- |
| cats | M2 ONNX single 640 | 5 | 9 | **+4** | 238 ms → 235 ms |
| cats | M3 ONNX 2×2 + Point-NMS | 14 | 14 | 0 | 525 ms → 527 ms |
| apples | M2 ONNX single 640 | 324 | 213 | −111 | 203 ms → 183 ms |
| apples | M4 Gemini single-pass | 551 | 248 | −303 | 115.4 s → 24.2 s |
| tennis-ball | M2 ONNX single 640 | 41 | 11 | −30 | 451 ms → 339 ms |
| tennis-ball | M3 ONNX 2×2 + Point-NMS | 24 | 0 | −24 | 987 ms → 972 ms |

Run id apples/tennis: `run_d8b1a6fb4a272b1a`, `run_09a3a53c5911d9a7`,
`run_3d100b0a1b64654f`, `run_6873b408a531c975`; run cats ada di `ab_runs.json`.

## Hasil 2 — M6 DeepSeek single-pass (VLM, 3 repetisi, mean ± sd)

| Gambar | Baseline | Inverted | Delta mean |
| --- | --- | --- | --- |
| apples | 387.0 ± 174.3 (190–521) | 305.7 ± 72.0 (239–382) | −81.3 |
| containers | 61.0 ± 27.7 (44–93) | 35.3 ± 2.3 (34–38) | −25.7 |
| candy | 137.7 ± 30.4 (103–160) | 100.0 ± 2.6 (98–103) | −37.7 |
| blueberry | 116.7 ± 12.1 (103–126) | 95.7 ± 1.5 (94–97) | −21.0 |
| coins | 24.0 ± 0.0 (24–24) | 24.0 ± 0.0 (24–24) | 0.0 |

## Observasi

1. **Tidak ada perbaikan yang konsisten.** Dari 11 sel uji, hanya cats/M2 yang naik
   (5 → 9); sisanya turun atau identik. Baik detektor offline (pack dilatih pada warna
   asli) maupun VLM kehilangan deteksi saat input dibalik.
2. **Offline paling rentan.** Pack apple/tennis kehilangan 34–73% deteksi; M3 pada
   tennis-ball jatuh ke 0. Pack tidak pernah melihat distribusi warna terbalik.
3. **VLM juga turun, tapi variasinya menyusut.** M6 kehilangan 17–42% pada apples,
   containers, candy, blueberry; containers turun dari 61 ± 28 menjadi 35 ± 2 (lebih
   stabil). coins tetap 24 di kedua kondisi — satu-satunya kelas yang tidak terpengaruh.
4. **Persepsi "lebih jelas" tidak terbawa ke model.** Mata melihat objek lebih kontras
   setelah inversi, tetapi hitungan model justru lebih sedikit. Baseline apples/M4 (551)
   juga jauh di atas perkiraan visual dan butuh 115 detik, jadi angka VLM baseline pun
   belum bisa dianggap patokan.

## Hasil 3 — citra baru dengan GT dari user (20 apel)

Foto apel padat (736×981, `img_a327a9b39997f70d`) diunggah ke playground dan dihitung
semua metode. GT = 20, dihitung manual oleh user (bukan pseudo-label) — ini datapoint
pertama dengan GT terverifikasi.

| Metode | Count | Selisih vs GT 20 | Waktu |
| --- | --- | --- | --- |
| M1 SAM3 | 27 | +7 | 24.3 s |
| M2 ONNX single | 35 | +15 | 223 ms |
| M3 ONNX 2×2 | 10 | −10 | 542 ms |
| M4 Gemini single | **20** | 0 | 9.2 s |
| M5 Gemini 2×2 | 30 | +10 | 9.0 s |
| M6 DeepSeek single | 22 | +2 | 3.0 s |
| M7 DeepSeek 2×2 | 28 | +8 | 2.1 s |

Run normal: `run_47d858bbfe7aa1bc`. Pada citra yang sama dengan `invert:true`
(`run_a4818014ee44a1b5`): M2 11, M3 0, M6 **20** (M6 kebetulan tepat, tetapi M2/M3
runtuh seperti pola spike 005/006 — bukan bukti perbaikan).

## Batasan

- Ground truth belum dihitung ulang manual; angka GT di benchmark (coins 24,
  containers 64, blueberry 113, apples ~289) hanya pembanding kasar, sehingga tabel ini
  membandingkan **count**, bukan akurasi.
- VLM sampling bervariasi besar (apples M6 190–521); n=3 cukup untuk melihat arah, bukan
  untuk klaim signifikansi.
- Kelas offline terbatas pada tiga pack yang ada; kelas lain tidak bisa diuji di M2/M3.

## Rekomendasi

Toggle `Invert warna` tetap opt-in sebagai alat eksperimen, **bukan** default. Kalau
riset dilanjutkan, dua arah yang masuk akal: (a) latih pack khusus citra inverted lalu
bandingkan dengan pack normal, atau (b) uji preprocessing lain yang tidak menggeser
distribusi warna (grayscale, CLAHE) dengan protokol dan GT manual yang sama.
