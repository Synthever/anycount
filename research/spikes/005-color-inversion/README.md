# Spike 005: Inversi Warna (RGB Negative) sebagai Preprocessing Counting

Pertanyaan: apakah membalik warna gambar (`R,G,B → 255−nilai`) sebelum counting membuat
objek lebih mudah dideteksi oleh ketujuh metode AnyCount?

Fitur yang diuji: flag `invert` pada `POST /api/runs` (lihat `prototype/README.md`).
Derivatif dibuat dari `work_path` (JPEG ≤1280px q90) sehingga satu-satunya variabel yang
berubah adalah warna, bukan resolusi atau kompresi. Semua metode dan renderer membaca
derivatif yang sama; `GET /api/runs/{id}/raw` menyajikan citra yang benar-benar dihitung.

## Setup

- Service: `anycount-prototype.service` (release build, `127.0.0.1:2030`, live di
  `https://testing.rkhyg.xyz`), dijalankan 2026-11-09.
- Sampel: `apples.jpg` (query `apples`), `tennis-ball.jpg` (query `tennis ball`).
- Baseline dan run inverted dieksekusi berurutan pada service yang sama; baseline
  memakai `invert:false`, perlakuan memakai `invert:true`.

## Hasil (jumlah objek terhitung)

| Gambar | Metode | Baseline | Inverted | Selisih | Waktu baseline → inverted |
| --- | --- | --- | --- | --- | --- |
| apples | M2 ONNX single 640 | 324 | 213 | −111 | 203 ms → 183 ms |
| apples | M4 Gemini single-pass | 551 | 248 | −303 | 115.4 s → 24.2 s |
| tennis-ball | M2 ONNX single 640 | 41 | 11 | −30 | 451 ms → 339 ms |
| tennis-ball | M3 ONNX 2×2 + Point-NMS | 24 | 0 | −24 | 987 ms → 972 ms |

Run id: `run_d8b1a6fb4a272b1a` (apples baseline), `run_09a3a53c5911d9a7` (apples inverted),
`run_3d100b0a1b64654f` (tennis baseline), `run_6873b408a531c975` (tennis inverted).

## Observasi

1. **Inversi konsisten menurunkan jumlah deteksi offline.** M2 kehilangan 34% (apples)
   dan 73% (tennis-ball); M3 pada tennis-ball jatuh ke 0 — pack dilatih pada warna asli,
   jadi membalik warna membuat distribusi input keluar dari data latih.
2. **VLM juga turun.** M4 pada apples 551 → 248. Baseline 551 sendiri jauh di atas
   perkiraan visual dan butuh 115 detik (antrean gateway), sedangkan run inverted selesai
   24 detik; angka 248 lebih dekat ke ~289 tetapi belum bisa disebut akurasi karena GT
   belum diverifikasi manual.
3. **Belum ada bukti "lebih keliatan" berbanding lurus dengan hasil hitung.** Persepsi
   mata bahwa objek lebih jelas setelah inversi tidak terbawa ke model: baik detektor
   terlatih maupun VLM justru menghasilkan hitungan lebih sedikit.

## Batasan

- Ground truth gambar belum dihitung ulang secara manual (angka 289 pada `apples` masih
  warisan dataset), jadi tabel ini membandingkan **count**, bukan akurasi.
- Satu run per kombinasi; variasi sampling VLM dan antrean gateway belum dirata-rata.
- Hanya dua gambar uji; kelas lain (coins, containers, Heidelbeeren) belum diuji.

## Rekomendasi

Toggle `Invert warna` tetap dipertahankan sebagai alat eksperimen, **tidak** dijadikan
default. Langkah riset berikutnya: ulangi A/B pada 3–4 sampel lain dengan 3 repetisi per
kombinasi, dan hitung GT manual untuk sampel yang dipakai agar selisih bisa dibaca
sebagai error, bukan sekadar count.
