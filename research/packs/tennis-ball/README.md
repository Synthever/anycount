# Tennis Ball Detector Pack (M2/M3 offline ONNX)

Paket bobot INT8 untuk pencarian objek `tennis ball` pada metode offline (M2 single-pass, M3 2x2 tiled).

## Cara dibuat

1. **Dataset publik.** 59 foto bola tenis diambil dari Wikimedia Commons lewat API
   (`generator=search`, `filetype:bitmap`), hanya lisensi CC0 / CC BY / CC BY-SA /
   public domain, resolusi >= 1000 px. Daftar lengkap judul, lisensi, pemilik, dan
   URL halaman ada di `manifest.json`. Berkas mentah disimpan di `raw/` dan
   **tidak** di-commit (lihat `.gitignore`).
2. **Label guru (LLM).** Tiap foto diberi titik oleh `gemini-3.8-flash-high`
   (prompt Count-Anything `[y, x]` skala 0..1000). Foto yang titiknya < 8 dijalankan
   ulang dengan 2x2 tiled + core-region filtering + Point-NMS. Hasil per foto ada di
   `labels/*.json` (`count`, `mode`, `points`).
3. **Kotak pseudo.** Ukuran kotak tiap titik = median jarak 3 tetangga terdekat x 0.92
   (maksimum 30% sisi pendek), sama seperti riset Metode 2 untuk apel.
4. **Dataset YOLO.** Foto utuh + 4 crop 2x2 (overlap 8%) + 28 komposit sintetis
   (patch bola asli ditempel 35-70 kali pada latar foto asli) = 239 gambar latih, 9 validasi.
5. **Latih.** `yolo11n.pt`, 35 epoch, `imgsz=640`, `batch=8`, `lr0=0.005`, `mosaic=0.5`,
   `close_mosaic=10`, CPU 6 core (sekitar 8 menit).
6. **Ekspor.** ONNX opset 12 `imgsz=640` lalu kuantisasi dinamis INT8.

## Hasil terukur

| Metrik | Nilai |
|---|---|
| Ukuran model | 3.03 MB (INT8) |
| mAP50 pada label pseudo (val 9 gambar) | 0.064 |
| Hitungan pada `research/samples/tennis-ball.jpg` (GT 43), M2 single | **41** (MAPE 4.7%) |
| Hitungan sama, M3 2x2 tiled | 26 (MAPE 39.5%) |

Catatan jujur: mAP rendah karena kotak validasi berasal dari pseudo-label yang kasar,
bukan anotasi manusia; metrik yang penting untuk aplikasi ini adalah selisih hitungan,
dan di situ mode single-pass sudah dekat dengan ground truth. Mode 2x2 tiled justru
lebih buruk pada foto bola yang renggang, karena crop 640 membuat bola kecil di sisi
panjang gambar makin kecil.

## Melatih ulang

```bash
/root/workspace/countthis/pilot/.venv/bin/python train_tennis.py          # bangun dataset
/root/workspace/countthis/pilot/.venv/bin/python train_tennis.py --train  # latih + ekspor INT8
```

Model hasil ekspor ditulis ke `models/tennis_ball_v1_int8.onnx` dan otomatis di-seed
sebagai pack `tennis-ball` saat server start (`seed_bundled_object_packs`).

## Lisensi gambar

Hanya berkas berlisensi bebas yang dipakai; atribusi per gambar tercatat di
`manifest.json` (judul, lisensi, pemilik, URL halaman Commons). Jika paket ini
dipakai di luar prototipe riset, atribusi CC BY / CC BY-SA wajib ditampilkan.
