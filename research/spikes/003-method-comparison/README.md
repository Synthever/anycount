# Spike 003: Side-by-Side Visual & Quantitative Comparison

Folder ini berisi skrip `compare_all.py` dan gambar montage perbandingan berdampingan (*side-by-side*) antara:
1. **Count-Anything (SAM3 3.4 GB)**
2. **Gemini 3.8 Flash — Single-Pass (`001`)**
3. **Gemini 3.8 Flash — 2x2 Tiled Grid + Point-NMS (`002`)**

## Daftar Gambar Montage (`results/`)

1. **`results/montage_cats_and_remotes.jpg`**
   - Membandingkan hasil Count-Anything (`cats: 2`, 23s di CPU) vs Gemini Single-Pass (`cats: 2`, 5.7s) vs kemampuan Open-Vocabulary Gemini pada gambar yang sama (`remote control: 2`, 5.0s).
2. **`results/montage_coins.jpg`**
   - Membandingkan Gemini Single-Pass (`24/24` koin) vs Gemini 2x2 Tiled + NMS (`24/24` koin, lengkap dengan garis batas tile tipis untuk memverifikasi tidak ada koin yang terhitung ganda di garis potong).
3. **`results/montage_containers.jpg`**
   - Bukti visual paling krusial:
     - Panel Kiri: **Count-Anything SAM3** (`container: 64`, 28.1s di Colab A100).
     - Panel Tengah: **Gemini Single-Pass** (`container: 29`, gagal mendeteksi kontainer sempit di tumpukan kiri dan belakang).
     - Panel Kanan: **Gemini 2x2 Tiled + NMS** (`container: 65`, berhasil menandai seluruh tumpukan kiri, tengah, dan kanan dengan akurasi 98.4% terhadap Count-Anything).
4. **`results/montage_apples.jpg`**
   - Membandingkan hasil pada tumpukan apel padat antara Count-Anything (`339`), Gemini Single-Pass (`289`), dan Gemini 2x2 Tiled (`343`).
