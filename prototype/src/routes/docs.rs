use axum::Json;
use serde_json::{json, Value};

pub async fn get_docs() -> Json<Value> {
    Json(json!({
        "methods": [
            {
                "id": "m1",
                "code": "M1",
                "name": "Count-Anything (SAM3 3.4 GB)",
                "badge": "Heavy Baseline",
                "color": "ochre",
                "speed": "14s - 28s",
                "footprint": "3.4 GB Checkpoint (7.2 GB RAM)",
                "summary": "Model riset berbasis Segment Anything 3 (848M parameter). Menempatkan titik di pusat mask tiap benda, namun sangat berat di RAM/GPU dan sensitif terhadap bayangan kontras tinggi.",
                "bestFor": "Pembanding laboratorium untuk foto warna standar."
            },
            {
                "id": "m2",
                "code": "M2",
                "name": "Offline ONNX Pack Single-Pass (640)",
                "badge": "On-Device Fast Path",
                "color": "mint",
                "speed": "~0.15s sparse · single 640 (Offline)",
                "footprint": "2.89 MB per objek (.onnx INT8)",
                "summary": "Aplikasi mengunduh bobot ringan .onnx sesuai objek yang dicari (mis. apel 2.89 MB) lalu menghitung dalam satu lintasan 640 tanpa internet. Pada foto uji apples (GT 289) jalur single terukur 324 titik (+12.1% dari GT) dalam ~0.15 s — satu sampel uji, bukan jaminan umum. Tidak ada eskalasi otomatis ke 2x2: mode padat adalah metode terpisah (M3).",
                "bestFor": "Foto jarang dengan kategori benda populer yang sering dihitung berulang kali, biaya server Rp 0."
            },
            {
                "id": "m3",
                "code": "M3",
                "name": "Offline ONNX Pack 2x2 Tiled + Point-NMS",
                "badge": "Dense Offline",
                "color": "teal",
                "speed": "~0.54s dense · tiled 2x2 (Offline)",
                "footprint": "2.89 MB per objek (.onnx INT8)",
                "summary": "Pack .onnx yang sama, tetapi selalu dijalankan sebagai empat petak 2x2 (overlap 8%) lalu digabung dengan Point-NMS. Pada foto uji apples (GT 289) terukur 289/289 titik (MAPE 0%, ~0.54 s) — angka satu sampel, bukan jaminan umum.",
                "bestFor": "Tumpukan padat ratusan objek yang butuh hitungan offline tanpa biaya server."
            },
            {
                "id": "m4",
                "code": "M4",
                "name": "Gemini 3.8 Flash Single-Pass",
                "badge": "Best Grounding",
                "color": "pink",
                "speed": "5s - 19s",
                "footprint": "0 MB (Cloud API)",
                "summary": "Mengirim foto utuh 1x ke gemini-3.8-flash-high untuk mendapatkan koordinat [y, x] tiap benda. Penempatan titik per-objek paling jujur dan presisi untuk foto jarak dekat hingga 130 benda.",
                "bestFor": "Pencarian kata bebas (open-vocabulary) foto jarak dekat & Validator pembuat label training M2/M3."
            },
            {
                "id": "m5",
                "code": "M5",
                "name": "Gemini 3.8 Flash 2x2 Tiled + NMS",
                "badge": "High-Detail Dense",
                "color": "lavender",
                "speed": "11s - 31s",
                "footprint": "0 MB (4x Parallel Cloud)",
                "summary": "Memotong gambar menjadi 4 petak (overlap 8%), memanggil Gemini paralel, dan menyatukan perbatasan dengan Adaptive Point-NMS. Mengangkat akurasi kontainer jauh dari 29 menjadi 65/64. Metode ini juga menjadi Guru Auto-Train pack .onnx.",
                "bestFor": "Objek kecil/jarak jauh (kontainer pelabuhan) dan foto minim warna/gelap."
            },
            {
                "id": "m6",
                "code": "M6",
                "name": "DeepSeek V4.1 Flash Single-Pass",
                "badge": "Fast Sparse",
                "color": "peach",
                "speed": "2.9s - 5.7s",
                "footprint": "0 MB (Cloud API)",
                "summary": "Menggunakan pola prompt Count-Then-Locate [x, y] pada deepseek-v4.1-flash-1017 tanpa jeda thinking panjang. Sangat cepat untuk jumlah benda <= 45.",
                "bestFor": "Hitung cepat objek sedikit-menengah (kucing, koin, bola tenis)."
            },
            {
                "id": "m7",
                "code": "M7",
                "name": "DeepSeek V4.1 Flash 2x2 Tiled + NMS",
                "badge": "Fastest Dense Cloud",
                "color": "emerald",
                "speed": "2.5s - 5.6s (Avg 3.79s)",
                "footprint": "0 MB (4x Parallel Cloud)",
                "summary": "Memadukan kecepatan DeepSeek V4.1 Flash dengan pembagian 2x2 paralel + Point-NMS. Menghilangkan gejala loop pada ratusan apel (336 apel dalam 5.6 detik).",
                "bestFor": "Mode hitung cepat di cloud untuk tumpukan padat ratusan benda."
            }
        ],
        "glossary": [
            {"term": "LLM / VLM (Vision-Language Model)", "plain": "Otak AI pintar di cloud (seperti Gemini dan DeepSeek) yang bisa melihat foto sekaligus memahami perintah teks bebas manusia."},
            {"term": "Open-Vocabulary", "plain": "Kemampuan mencari benda apa saja cukup dengan mengetik namanya tanpa dibatasi pilihan menu kategori yang kaku."},
            {"term": "Spatial Pointing [0..1000]", "plain": "Cara AI menunjuk letak pusat benda memakai skala angka seribu dari kiri ke kanan dan dari atas ke bawah."},
            {"term": "Single-Pass (Foto Utuh)", "plain": "Mengirim satu lembar foto utuh dalam sekali tanya tanpa dipotong-potong."},
            {"term": "Tiled Grid 2x2 (Petak 2x2)", "plain": "Memotong foto menjadi 4 kuadran seperti kaca pembesar agar benda mungil di kejauhan terlihat 2x lebih jelas oleh AI."},
            {"term": "Overlap 8%", "plain": "Lebihan area potong di garis tengah supaya benda yang kebetulan berada pas di garis potong tidak terbelah dua."},
            {"term": "Point-NMS (Penghapus Titik Kembar)", "plain": "Saringan otomatis yang menggabungkan dua titik berdekatan di garis sambungan petak agar satu benda tidak terhitung dua kali."},
            {"term": "ONNX & Kuantisasi INT8", "plain": "Format file model AI ringkas untuk HP/komputer tanpa internet; INT8 memadatkan ukuran model dari 10 MB menjadi ~2.89 MB."},
            {"term": "Cold-Start (Objek Baru di M2/M3)", "plain": "Kondisi saat pengguna mencari benda yang belum punya file .onnx lokal, sehingga sistem meminta bantuan LLM terlebih dahulu."},
            {"term": "Distillation Flywheel (Guru ke Murid)", "plain": "Siklus otomatis di mana titik jawaban dari AI besar (Guru LLM) dipakai melatih model kecil (.onnx) agar ikut menjadi pintar."},
            {"term": "Ground Truth (GT) & MAPE", "plain": "GT adalah jumlah benda asli hasil hitungan teliti manusia; MAPE adalah persentase rata-rata melesetnya hitungan AI (makin mendekati 0% makin akurat)."},
            {"term": "False Positive & False Negative", "plain": "False Positive adalah salah menaruh titik di tempat bukan benda; False Negative adalah benda asli yang terlewat belum ditandai."}
        ]
    }))
}
