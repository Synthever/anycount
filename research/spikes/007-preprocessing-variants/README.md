# Spike 007: Grayscale dan CLAHE sebagai Preprocessing Counting

Pertanyaan: apakah preprocessing selain inversi — grayscale dan CLAHE — memperbaiki
hitungan dibandingkan citra asli, pada protokol yang sama dengan spike 005/006?

## Protokol

- Input: 9 gambar validasi tennis-ball (ref = pseudo-label, rata-rata 3.8 objek) +
  `research/samples/tennis-ball.jpg` (ref 43, angka benchmark, belum diverifikasi manual).
- Varian: `normal`, `gray` (luminance → RGB), `clahe` (CLAHE clip 2.0, tile 8×8 pada
  kanal L di ruang LAB, warna dipertahankan).
- Metode: M2/M3 memakai pack tennis-ball yang shipped (`run_pass`, `run_tiled_pass`);
  M6 DeepSeek single-pass lewat harness spike 005 (temperature 0.1).
- Data: `variants.json`. Reproduksi: `python3 run_variants.py` (venv `countthis/pilot/.venv`).

## Hasil — sampel padat `tennis-ball.jpg` (ref 43)

| Varian | M2 single | M3 2×2 | M6 DeepSeek |
| --- | --- | --- | --- |
| normal | **41** | **26** | 41 |
| gray | 1 | 0 | 42 |
| clahe | 40 | 14 | 41 |

## Hasil — 9 gambar validasi (mean count / MAPE vs pseudo-label)

| Varian | M2 | M3 | M6 |
| --- | --- | --- | --- |
| normal | 4.8 / 72.5% | 2.9 / 82.0% | 5.2 / 40.6% |
| gray | 0.3 / 95.8% | 0.0 / 100% | 5.6 / 37.4% |
| clahe | 4.5 / 83.7% | 1.9 / 78.7% | 5.6 / 47.6% |

## Observasi

1. **Grayscale mematikan detektor offline.** M2 pada sampel padat jatuh 41 → 1; pack
   dilatih pada citra berwarna, dan warna bola adalah sinyal utama. VLM tidak terpengaruh
   (41 → 42, selisih satu objek = noise sampling).
2. **CLAHE hampir netral.** M2 41 → 40 (tidak berbeda), M3 26 → 14 (lebih buruk), M6
   41 → 41. Pada validasi renggang, MAPE M3 justru membaik tipis (82% → 79%) tetapi M6
   memburuk (40.6% → 47.6%); tidak ada arah yang konsisten.
3. **Tidak ada preprocessing yang mengalahkan citra asli.** Satu-satunya sel "menang"
   adalah M6 gray (42 vs 41) dan M6 val (MAPE 37% vs 40%) — keduanya dalam rentang variasi
   sampling VLM yang sudah terlihat di spike 005 (apples M6 190–521), bukan sinyal nyata.
4. **Validasi terlalu renggang.** Rata-rata 3.8 objek/gambar membuat MAPE sangat kasar;
   angka sampel padat lebih dapat dipercaya untuk arah.

## Kesimpulan

Untuk pipeline saat ini, preprocessing warna tidak memberi keuntungan: inversi butuh
pack khusus dan tetap kalah (spike 006), grayscale merusak pack offline, CLAHE netral.
Kalau riset dilanjutkan, arah yang lebih menjanjikan adalah perbaikan data/label
(GT manual, label lebih padat) atau arsitektur/model, bukan preprocessing warna.
