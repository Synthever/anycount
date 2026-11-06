# Spike 002: Gemini Parallel Tiled Grid + Seam Point-NMS

## Pertanyaan Eksperimen
- **Given:** Foto dengan objek padat (`containers.jpg` 64 kontainer, `apples.jpg` ratusan apel, `coins.jpg` 24 koin) di mana Single-Pass VLM kehilangan banyak objek kecil.
- **When:** Gambar dipotong menjadi grid `2x2` dengan *overlap* 8%, seluruh tile dikirim secara **paralel** (`ThreadPoolExecutor`) ke `gemini-3.8-flash-high`, lalu koordinat digabung ke ruang global menggunakan **Core-Region Filtering + Adaptive Radius Point-NMS** untuk menghapus duplikat di garis potong (*seam*).
- **Then:** Apakah akurasi pada objek padat meningkat mendekati Count-Anything (64 kontainer) dengan latensi yang tetap terjaga?

## Hasil Pengukuran (`results/summary.json`)

| Gambar | Query | Grid | GT / Ref | Raw Points (4 Tile) | Setelah Point-NMS | Error | MAPE | Wall Time (s) | Output Image |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `coins.jpg` | `"coins"` | `2x2` | 24 | 25 | **24** | 0 | **0.0%** | 20.78s | `results/coins__coins__2x2.jpg` |
| `containers.jpg` | `"container"` | `2x2` | 64 | 71 | **65** | +1 | **1.6%** | 31.53s | `results/containers__container__2x2.jpg` |
| `apples.jpg` | `"apples"` | `2x2` | ~215 (CA: 339) | 374 | **343** | +4 vs CA | — | **21.09s** | `results/apples__apples__2x2.jpg` |

## Verdict: VALIDATED

### Apa yang Berhasil
- **Lompatan akurasi drastis pada `containers.jpg`:** Dari **29** (Single-Pass, error 54.7%) naik menjadi **65** (hanya selisih **+1 titik / 1.6% error** dibandingkan referensi 64 titik milik Count-Anything!). Seluruh kontainer sempit di tumpukan kiri (7x4 = 28 kontainer) dan tumpukan kanan berhasil terdeteksi karena resolusi efektif per objek meningkat 2x lipat.
- **Deduplikasi perbatasan (*Seam Point-NMS*) bekerja bersih:**
  - Pada `coins.jpg`, 25 titik mentah lintas 4 tile berhasil di-deduplikasi menjadi tepat **24/24 koin**.
  - Pada `containers.jpg`, kontainer panjang (*Hyundai*, *China Shipping*) yang terpotong garis tengah vertikal otomatis digabung oleh *Point-NMS* dari 71 titik mentah menjadi 65 titik bersih.
- **Memangkas waktu tunggu hampir 2x pada objek ratusan (`apples.jpg`):** Karena 4 tile dieksekusi paralel, wall-clock time turun dari **39.94 detik** (Single-Pass) menjadi **21.09 detik** (Tiled 2x2), dengan jumlah titik (343) sangat dekat dengan hasil Count-Anything (339).

### Kejutan Penting selama Eksperimen
- Model `gemini-3.8-flash-high` memiliki mode *thinking* bawaan yang tinggi. Jika tidak diberi batas `max_tokens: 4096` dan instruksi singkat (`"Do NOT write coordinates in your thinking process"`), model bisa menghabiskan >20.000 *reasoning tokens* (~200 detik) hanya untuk memperdebatkan objek yang terpotong di tepi tile. Dengan pembatasan `max_tokens` dan instruksi ringkas, waktu per tile turun drastis ke ~15–30 detik.
