# AnyCount Web Prototype (`prototype/`) — Design Specification

- **Status:** Draft v1 — Menunggu review pengguna sebelum pembuatan rencana implementasi (`writing-plans`)
- **Tanggal:** 2026-11-07
- **Target Lokasi Repo:** `prototype/` (di `D:\Github\anycount\prototype\` dan `/root/workspace/anycount/prototype/`)
- **Target Domain Produksi:** `https://testing.rkhyg.xyz` (`49.128.191.225`, di-proxy oleh Caddy ke `127.0.0.1:2030`)

---

## 1. Tujuan & Cakupan

Membangun aplikasi web prototipe interaktif **AnyCount Studio** di dalam direktori `prototype/` yang memungkinkan pengguna dan tim penguji mencoba secara langsung keenam metode penghitungan objek hasil riset AnyCount, membandingkan akurasi serta kecepatan antar-metode secara berdampingan (*Compare Mode*), menguji siklus *Cold-Start* dan *Auto-Train* pada paket model offline `.onnx`, serta merasakan pengalaman koreksi titik manual di atas kanvas interaktif sebagaimana gambaran sistem aplikasi AnyCount nantinya.

### Kriteria Keberhasilan
1. Seluruh **6 metode** (`M1` Count-Anything SAM3, `M2` Offline ONNX + Validasi/Auto-Train LLM, `M3` Gemini 3.8 Flash Single-Pass, `M4` Gemini 3.8 Flash 2x2 Tiled + NMS, `M5` DeepSeek V4.1 Flash Single-Pass, `M6` DeepSeek V4.1 Flash 2x2 Tiled + NMS) dapat dieksekusi dari satu antarmuka web terpadu.
2. Mendukung mode **Single Method** maupun **Side-by-Side Compare Mode** dengan pembaruan progres *real-time* via *Server-Sent Events (SSE)* per metode (metode yang selesai dalam 0.15 detik atau 3.8 detik langsung merender hasilnya tanpa menunggu metode yang berjalan 20 detik).
3. Menyediakan **Interactive Dot Canvas** di setiap kartu hasil: pengguna dapat mengklik area kosong untuk menambah titik (`+1`), mengklik titik merah untuk menghapusnya (`-1`), melakukan `Undo`/`Reset`, menyalakan/mematikan garis bantu petak `2x2`, mengintip `Raw JSON`, dan mengunduh gambar JPG hasil anotasi bergaya Count-Anything.
4. Antarmuka mengadopsi tata letak **JRouter Playground** (`docs/design/image.png`) dengan mode gelap sebagai tampilan awal (*dark-first* `#141414`) serta tombol pengalih (*Theme Toggle*) ke mode terang krem hangat (`#fffaf0`) beserta palet kartu aksen dari `JRouter/docs/DESIGN.md`.
5. Dilindungi oleh gerbang kode akses ringan (**default passcode: `2limanov`**) untuk eksekusi *live* agar kuota API dan memori VPS aman, sementara galeri sampel hasil riset, katalog model `.onnx`, dan halaman dokumentasi tetap dapat dilihat langsung saat halaman dibuka.
6. Berjalan stabil di VPS `49.128.191.225` pada `127.0.0.1:2030` di balik HTTPS Caddy `https://testing.rkhyg.xyz` dan terdaftar resmi di database `sites` milik `vpsdash` sehingga konfigurasi tidak tertimpa.

---

## 2. Fakta Lingkungan Host (Hasil Probe Nyata di VPS)

Seluruh keputusan desain di bawah didasarkan pada hasil pengecekan langsung di VPS `49.128.191.225`:
- **DNS:** `testing.rkhyg.xyz` telah mengarah ke `49.128.191.225`.
- **Resource VPS:** 6 vCPU AMD EPYC-Genoa, 14 GB RAM (~12 GB *available*), 91 GB ruang kosong pada `/`. Tidak ada swap aktif (`Swap: 0B`), sehingga eksekusi `Count-Anything` (~7.2 GB puncak RSS) **wajib** dibatasi maksimal 1 proses serentak dengan `Semaphore::new(1)`.
- **Toolchain Tersedia:** `rustc 1.99.1`, `cargo 1.99.1`, `node v26.13.0` & `npm 11.20.0` (di bawah `/root/.nvm/versions/node/v26.13.0/bin`), Python 3.12 venv `/root/workspace/count-anything/.venv` (berisi PyTorch CPU + SAM3 Count-Anything), dan `/root/workspace/countthis/pilot/.venv` (berisi `onnxruntime 1.31.1` + `ultralytics 8.4.202`).
- **Runtime & Supervisor:** Docker tidak terpasang di VPS ini (`NO_DOCKER`). Servis Rust produksi eksisting (`vpsdash` di `127.0.0.1:2020`) dijalankan langsung melalui unit `systemd`. Oleh karena itu, prototipe akan dideploy aktif di VPS menggunakan unit `systemd` (`anycount-prototype.service` di `127.0.0.1:2030`), sembari tetap menyertakan `Dockerfile` dan `compose.yml` di `prototype/` untuk portabilitas kontainer di mesin lain.
- **Manajemen Caddy oleh `vpsdash`:** File `/etc/caddy/Caddyfile` dikelola otomatis oleh `vpsdash` (database SQLite `/root/.local/share/vpsdash/vpsdash.db`, tabel `sites` dan `caddy_state`). Pendaftaran `testing.rkhyg.xyz -> 127.0.0.1:2030` wajib dimasukkan ke tabel `sites` pada `vpsdash.db`, dirender mengikuti format `vpsdash`, memperbarui hash SHA-256 di `caddy_state`, lalu divalidasi (`caddy validate --config /etc/caddy/Caddyfile`) dan di-reload (`systemctl reload caddy`).

---

## 3. Arsitektur & Struktur Folder

Sistem dibangun sebagai **Monolitik Single-Binary Rust (Axum)** yang membundel aset statis SPA React (`web/dist`) menggunakan `rust-embed`.

```text
prototype/
├── Cargo.toml                        # Crate Rust backend (axum, tokio, sqlx, reqwest, image, rust-embed)
├── Cargo.lock
├── .env.example                      # Contoh variabel lingkungan tanpa kredensial rahasia
├── Dockerfile                        # Multi-stage build (Node build web/ -> Rust build -> Debian slim)
├── compose.yml                       # Konfigurasi Docker Compose portabel
├── deploy/
│   ├── anycount-prototype.service    # Unit systemd untuk VPS 49.128.191.225
│   ├── build.sh                      # Skrip build web/dist + cargo build --release
│   └── register_caddy_vpsdash.py     # Skrip idempoten registrasi testing.rkhyg.xyz ke vpsdash.db + Caddy
├── migrations/
│   └── 0001_init.sql                 # Skema SQLite: sessions, images, runs, run_methods, model_packs
├── workers/
│   ├── ca_worker.py                  # Worker Python tipis untuk Metode 1 (Count-Anything SAM3)
│   └── onnx_worker.py                # Worker Python untuk inferensi ONNX cepat & Auto-Train pack baru
├── src/
│   ├── main.rs                       # Entrypoint server Axum, migrasi DB, seed sampel & pack apel
│   ├── config.rs                     # Pembacaan dan validasi env vars
│   ├── state.rs                      # AppState (SqlitePool, Semaphore(1), broadcast SSE channels)
│   ├── error.rs                      # AppError terstruktur -> JSON HTTP response
│   ├── auth.rs                       # Verifikasi passcode (default "2limanov") & cookie sesi HttpOnly
│   ├── routes/
│   │   ├── mod.rs
│   │   ├── auth.rs                   # GET /api/auth/status, POST /api/auth/unlock, POST /api/auth/logout
│   │   ├── images.rs                 # POST /api/images (upload), GET /api/images/:id/raw
│   │   ├── samples.rs                # GET /api/samples (11 skenario benchmark siap pakai)
│   │   ├── runs.rs                   # POST /api/runs, GET /api/runs, GET /api/runs/:id, GET /api/runs/:id/stream
│   │   ├── packs.rs                  # GET /api/packs, POST /api/packs/train, GET /api/packs/:slug/download
│   │   └── docs.rs                   # GET /api/docs (ringkasan 6 metode & kamus istilah awam)
│   └── services/
│       ├── mod.rs
│       ├── vlm.rs                    # Klien native Rust untuk M3, M4 (Gemini) & M5, M6 (DeepSeek)
│       ├── tiling.rs                 # Pemotongan gambar 2x2 (overlap 8%) + Core-Region + Adaptive Point-NMS
│       ├── renderer.rs               # Pelukis titik merah-putih + label pojok kiri atas (gaya Count-Anything)
│       ├── m1_ca.rs                  # Eksekutor M1 via ca_worker.py di bawah Semaphore(1)
│       └── m2_onnx.rs                # Eksekutor M2 ONNX (inferensi, cek Cold-Start, & job Auto-Train)
└── web/
    ├── package.json
    ├── tsconfig.json
    ├── vite.config.ts
    ├── index.html
    └── src/
        ├── main.tsx
        ├── App.tsx
        ├── styles/
        │   └── tokens.css            # CSS custom properties dari JRouter DESIGN.md (dark & warm-cream light)
        ├── api/
        │   └── client.ts             # Wrapper fetch + EventSource SSE listener
        ├── components/
        │   ├── Sidebar.tsx           # Navigasi kiri bergaya JRouter Proxy (Studio, Benchmark, Packs, Docs)
        │   ├── Topbar.tsx            # Breadcrumb, Theme Toggle, status Passcode Unlock
        │   ├── Composer.tsx          # Upload/drop gambar, Quick Sample chips, input query, pemilih M1-M6
        │   ├── MethodResultCard.tsx  # Kartu hasil per metode + telemetri monospace + aksi (Edit, Grid, Raw)
        │   ├── InteractiveCanvas.tsx # Kanvas interaktif klik tambah/hapus titik + Undo/Reset + ekspor JPG
        │   └── PasscodeModal.tsx     # Dialog ringkas input passcode (default: 2limanov) saat menjalankan live test
        └── pages/
            ├── PlaygroundPage.tsx    # Halaman utama uji coba Single & Side-by-Side Compare
            ├── BenchmarkPage.tsx     # Galeri instan 11 skenario hasil riset & montage perbandingan
            ├── ModelPacksPage.tsx    # Katalog model .onnx Metode 2, ukuran MB, & pemicu Auto-Train
            └── DocsGlossaryPage.tsx  # Penjelasan cara kerja M1-M6 & kamus istilah awam
```

---

## 4. Skema Data (SQLite)

Disimpan di `ANYCOUNT_DATA_DIR/anycount.db` (default `/var/lib/anycount-prototype/anycount.db`):

1. **`sessions`**: `token_hash TEXT PRIMARY KEY`, `created_at TEXT`, `expires_at TEXT` (masa berlaku 30 hari setelah input passcode valid).
2. **`images`**: `id TEXT PRIMARY KEY`, `filename TEXT`, `width INTEGER`, `height INTEGER`, `orig_path TEXT`, `work_path TEXT` (sisi panjang maks `1280px` JPEG quality 90), `is_sample INTEGER`, `created_at TEXT`.
3. **`runs`**: `id TEXT PRIMARY KEY`, `image_id TEXT`, `query TEXT`, `status TEXT` (`running` | `completed` | `partial`), `created_at TEXT`.
4. **`run_methods`**:
   - `run_id TEXT`, `method_id TEXT` (`m1`..`m6`), `status TEXT` (`queued` | `running` | `completed` | `cold_start` | `failed`)
   - `count INTEGER`, `points_json TEXT` (`[{"x": f64, "y": f64}]` dalam koordinat piksel gambar asli)
   - `grid_boxes_json TEXT` (koordinat garis petak `2x2` untuk `m4` dan `m6`, opsional pada `m2` padat)
   - `rendered_path TEXT` (path JPG hasil anotasi titik merah-putih statis)
   - `elapsed_ms INTEGER`, `prompt_tokens INTEGER`, `completion_tokens INTEGER`, `model_size_mb REAL`
   - `meta_json TEXT` (detail mentah/raw response, info mode single vs tiled, atau alasan cold-start/error)
5. **`model_packs`**:
   - `slug TEXT PRIMARY KEY` (mis. `apple`), `display_name TEXT`, `aliases_json TEXT` (`["apple", "apples", "apel"]`), `onnx_path TEXT`, `size_mb REAL`, `imgsz INTEGER`, `trained_epochs INTEGER`, `val_mape_pct REAL`, `status TEXT` (`ready` | `training` | `failed`), `updated_at TEXT`.
   - Saat startup pertama, sistem otomatis mendaftarkan `apple_v2_distilled_int8.onnx` (`2.89 MB`, MAPE `2.33%`) dari hasil riset ke dalam katalog `model_packs` dengan status `ready`.

---

## 5. Spesifikasi Detail Enam Metode & Alur Eksekusi

### 5.1 Konvensi Koordinat & Renderer Seragam
- Seluruh koordinat dari ketujuh komponen diseragamkan di backend menjadi piksel gambar asli `(x, y)` dengan `0 <= x < width` dan `0 <= y < height`.
- `services/renderer.rs` menggambar lingkaran merah `rgba(230, 40, 35, 230)` dengan cincin tepi putih `rgba(255, 255, 255, 255)` berjari-jari `max(4, round(min(w, h) / 180))` serta kotak label hitam semi-transparan `<query>: <count>` di pojok kiri atas untuk file unduhan JPG, sementara di browser `InteractiveCanvas.tsx` menggambarnya secara dinamis di atas `<canvas>` sehingga responsif terhadap klik koreksi pengguna.

### 5.2 Perilaku Tiap Metode
- **Metode 1 (`m1` — Count-Anything SAM3 3.4 GB):**
  - Meminta izin dari `heavy_semaphore` (`Semaphore::new(1)`). Selama menunggu giliran, SSE memancarkan status `queued`.
  - Menjalankan `workers/ca_worker.py` menggunakan `/root/workspace/count-anything/.venv/bin/python` pada `work_path`, mengambil JSON prediksi titik, merender gambar, lalu melepas semaphore.
- **Metode 2 (`m2` — Offline Modular ONNX + LLM Validator):**
  - Mencocokkan `query` (case-insensitive, trim, singular/plural sederhana + pencocokan `aliases_json`) ke tabel `model_packs` yang berstatus `ready`.
  - **Jalur Pack Tersedia (mis. `apple` / `apples` / `apel`):** Menjalankan `workers/onnx_worker.py infer` dengan `/root/workspace/countthis/pilot/.venv/bin/python` (mode adaptif `640px` single-pass, otomatis lanjut ke `2x2 Tiled ONNX + Point-NMS` bila jumlah deteksi padat) yang selesai dalam ~140–220 ms. Di kartu hasil tersedia tombol **"Validate with LLM"** yang memanggil M3/M4 untuk menampilkan perbandingan skor Offline vs Validator LLM.
  - **Jalur Cold-Start (Pack Belum Tersedia, mis. `coins`, `blueberry`, `bottle`):** Mengembalikan `status: "cold_start"` secara instan (< 10 ms) tanpa error palsu. Kartu menampilkan penjelasan bahwa bobot `.onnx` untuk objek tersebut belum diunduh/dilatih, disertai dua tombol:
    1. **"Fallback ke LLM Validator"** (menjalankan M3/M4 pada gambar tersebut).
    2. **"Auto-Train Pack `.onnx` Baru"** (`POST /api/packs/train`): mengambil titik dari Validator LLM pada gambar tersebut, membangun *pseudo-label* kotak tetangga terdekat + augmentasi multi-skala di bawah `heavy_semaphore`, melatih `yolo11n` menjadi `<slug>_int8.onnx` (~2.9 MB), dan menyimpannya ke tabel `model_packs` sehingga pengujian berikutnya langsung berjalan via ONNX!
- **Metode 3 (`m3` — Gemini 3.8 Flash Single-Pass):**
  - Native Rust `reqwest` ke `VLM_API_URL` dengan model `GEMINI_MODEL` (`gemini-3.8-flash-high`), `reasoning_effort: "low"`, `max_tokens: 4096`, `temperature: 0.1`.
  - Mem-parsing *stream* SSE `data: {...}`, mengekstrak pasangan `[y, x]` (`0..1000`), dan memetakannya ke piksel `(x, y)`.
- **Metode 4 (`m4` — Gemini 3.8 Flash 2x2 Tiled Grid + NMS):**
  - Native Rust `services/tiling.rs` memotong gambar menjadi `2x2` kuadran dengan *overlap* `8%`, mengirim keempat kuadran secara paralel (`futures::future::join_all`), menyaring titik tepi luar (*Core-Region Filtering*), dan menjalankan **Adaptive Radius Point-NMS** (`radius = clamp(median_nn * 0.55, 8.0, min(w, h) * 0.06)`, faktor lintas-tile `1.35x`, dalam-tile `0.75x`).
- **Metode 5 (`m5` — DeepSeek V4.1 Flash Single-Pass):**
  - Native Rust `reqwest` dengan model `DEEPSEEK_MODEL` (`deepseek-v4.1-flash-1017`), `max_tokens: 8192`, `temperature: 0.1`, memakai prompt alami dua tahap (*Count-Then-Locate*).
  - Mem-parsing respons JSON + `data: [DONE]` dan memetakan pasangan koordinat `[x, y]` (`0..1000`) ke piksel `(x, y)`.
- **Metode 6 (`m6` — DeepSeek V4.1 Flash 2x2 Tiled Grid + NMS):**
  - Menggabungkan pemotongan paralel `2x2` pada `services/tiling.rs` dengan klien DeepSeek V4.1 Flash (`[x, y]`) serta deduplikasi *Adaptive Radius Point-NMS*.

---

## 6. Desain Antarmuka (Mengacu `docs/design/image.png` & `JRouter/docs/DESIGN.md`)

### 6.1 Sistem Tema Ganda (Dark Default + Warm-Cream Light Toggle)
- **Dark Mode (Default saat pertama dibuka, sesuai `docs/design/image.png` & `.dark` JRouter):**
  - Kanvas utama `#141414` / `#0e0e10`, kartu permukaan `#1c1c1c` / `#202020`, input/composer `#18181b`, garis batas tipis `rgba(255, 255, 255, 0.08)`, teks utama `#fffaf0`, teks sekunder `#adbdb5`, pill navigasi aktif putih hangat `#fffaf0` dengan teks `#0a0a0a`, aksen pengguna & toggle aktif lavender `#b8a4ed`, badge sukses emerald `#34d39a`, dan sorotan aksi pink `#ff4d8b`.
- **Light Mode (Sesuai `JRouter/docs/DESIGN.md`):**
  - Kanvas krem hangat `#fffaf0`, permukaan lembut `#faf5e8`, permukaan kartu `#f5f0e0`, batas hairline `#e5e5e5`, teks tinta `#0a0a0a`, tombol utama `#0a0a0a`, serta kartu fitur berpalet warna khas JRouter (`brand-pink #ff4d8b`, `brand-teal #1a3a3a`, `brand-lavender #b8a4ed`, `brand-peach #ffb084`, `brand-ochre #e8b94a`).
- **Tipografi:** `Inter` untuk UI dan teks isi (`weight 400–600`), headline bergaya display `weight 500` dengan *negative letter-spacing* (`-0.03em` s/d `-0.05em`), serta font monospace (`JetBrains Mono` / `ui-monospace`) untuk telemetri waktu, jumlah token, ukuran model, dan tampilan Raw JSON.

### 6.2 Empat Halaman Navigasi Sidebar
1. **Playground (`/`):**
   - Area tengah menampilkan kartu sesi pengujian (foto sumber & query di kanan atas bergaya bubble lavender `image.png`, diikuti satu kartu hasil lebar penuh pada *Single Mode* atau grid 2 kolom kartu hasil berdampingan pada *Compare Mode*).
   - **Interactive Dot Canvas** pada setiap kartu: mendukung zoom/fit, mode `Edit Dots` (klik tambah/hapus titik merah-putih secara langsung, penghitung angka di pojok gambar dan di footer ikut berubah seketika disertai penanda `(+N / -M manual)`), toggle `2x2 Grid Overlay`, tampilan `<> Raw` JSON, dan tombol `Download JPG`.
   - **Floating Composer** di bagian bawah layar: mendukung klik upload / *drag-and-drop* / tempel gambar dari *clipboard* (`Ctrl+V`), deretan *Quick Sample Pills* untuk mencoba foto sampel dalam 1 klik, kolom input kata benda yang dicari, pemilih metode (`M1`..`M6` satuan atau multi-select `Compare`), serta tombol pintas `Select Fast Cloud (M3+M6)` dan `Select All 6 Methods`.
2. **Benchmark Gallery (`/benchmarks`):**
   - Menampilkan ke-11 skenario hasil pengujian riset beserta gambar montage perbandingan yang dimuat langsung dari data riset tanpa harus menjalankan ulang inferensi (terbuka bebas tanpa perlu memasukkan passcode).
3. **Model Packs — Metode 2 (`/packs`):**
   - Menampilkan daftar paket bobot `.onnx` yang terpasang (dimulai dengan `apple` `2.89 MB`, akurasi MAPE `2.33%`), tombol unduh file `.onnx` untuk simulasi unduhan ke HP, serta formulir untuk melatih pack `.onnx` objek baru dari foto + Validator LLM.
4. **Method Docs & Glossary (`/docs`):**
   - Menyajikan ringkasan visual cara kerja keenam metode, panduan kapan memakai tiap metode, serta Kamus Istilah Teknis berbahasa Indonesia yang ramah pembaca awam.

---

## 7. Konfigurasi Variabel Lingkungan (`.env`)

Seluruh variabel konfigurasi dideklarasikan di `prototype/src/config.rs` dan `.env.example`:

| Variabel | Default (Non-Rahasia) | Keterangan & Bagian Pemakai |
|---|---|---|
| `ANYCOUNT_BIND` | `127.0.0.1:2030` | Alamat loopback server Axum (`main.rs` & upstream Caddy) |
| `ANYCOUNT_DATA_DIR` | `/var/lib/anycount-prototype` | Direktori SQLite, unggahan gambar, hasil render, dan pack `.onnx` (`state.rs`, `images.rs`, `packs.rs`) |
| `ANYCOUNT_PASSCODE` | `2limanov` | Kode akses untuk membuka eksekusi live di Playground & Auto-Train (`auth.rs`) |
| `ANYCOUNT_REPO_ROOT` | `/root/workspace/anycount` | Akar repositori untuk membaca gambar sampel & hasil benchmark (`samples.rs`, `m2_onnx.rs`) |
| `VLM_API_URL` | `https://jr.rkhyg.xyz/v1/chat/completions` | Endpoint gateway OpenAI-compatible untuk Metode 3, 4, 5, 6 (`services/vlm.rs`) |
| `VLM_API_KEY` | *(wajib diisi di `.env` VPS, kosong di git)* | Kunci API gateway VLM; hanya dibaca di server Rust & worker, tidak pernah dikirim ke klien (`services/vlm.rs`, `m2_onnx.rs`) |
| `GEMINI_MODEL` | `gemini-3.8-flash-high` | Nama model untuk Metode 3, Metode 4, dan Validator Metode 2 (`services/vlm.rs`) |
| `DEEPSEEK_MODEL` | `deepseek-v4.1-flash-1017` | Nama model untuk Metode 5 dan Metode 6 (`services/vlm.rs`) |
| `CA_PYTHON_BIN` | `/root/workspace/count-anything/.venv/bin/python` | Interpreter Python untuk worker Metode 1 (`services/m1_ca.rs`) |
| `CA_REPO_DIR` | `/root/workspace/count-anything` | Direktori kerja dan checkpoint `count_anything.pt` untuk Metode 1 (`services/m1_ca.rs`) |
| `ONNX_PYTHON_BIN` | `/root/workspace/countthis/pilot/.venv/bin/python` | Interpreter Python dengan `onnxruntime` & `ultralytics` untuk Metode 2 (`services/m2_onnx.rs`) |
| `YOLO_BASE_WEIGHTS` | `/root/workspace/countthis/pilot/yolo11n.pt` | Bobot dasar YOLO11n untuk job Auto-Train pack baru Metode 2 (`services/m2_onnx.rs`) |

---

## 8. Deployment & Integrasi Caddy + `vpsdash`

### 8.1 Unit `systemd` (`/etc/systemd/system/anycount-prototype.service`)
```ini
[Unit]
Description=AnyCount Prototype Web Studio (Rust Axum + React)
After=network-online.target caddy.service
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/workspace/anycount/prototype
EnvironmentFile=/root/workspace/anycount/prototype/.env
ExecStart=/root/workspace/anycount/prototype/target/release/anycount-prototype
Restart=always
RestartSec=2
MemoryMax=10G
LimitNOFILE=65535
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```
*(Catatan: `PrivateTmp` sengaja tidak diaktifkan dan `MemoryMax=10G` diberikan agar sub-proses `ca_worker.py` dan `onnx_worker.py` leluasa memuat `mmap` checkpoint 3.4 GB tanpa terbunuh cgroup kecil).*

### 8.2 Registrasi Domain `testing.rkhyg.xyz` Tanpa Konflik `vpsdash`
Skrip `prototype/deploy/register_caddy_vpsdash.py` melakukan langkah atomik berikut:
1. Membuka `/root/.local/share/vpsdash/vpsdash.db` menggunakan modul bawaan `sqlite3` Python 3.12 dan menyisipkan (`INSERT OR UPDATE`) baris pada tabel `sites`:
   - `host = 'testing.rkhyg.xyz'`, `type = 'proxy'`, `upstream = '127.0.0.1:2030'`, `enabled = 1`.
2. Membaca seluruh baris `enabled = 1` dari tabel `sites` berurutan abjad (`ORDER BY host`), merender teks `/etc/caddy/Caddyfile` persis dengan header `# managed by vpsdash — jangan edit manual, perubahan akan tertimpa`.
3. Menjalankan `/usr/bin/caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile`, menulis file secara atomik, memperbarui `rendered_hash` dan `applied_hash` (SHA-256 hex) pada tabel `caddy_state`, lalu menjalankan `systemctl reload caddy`.

---

## 9. Rencana Pengujian & Verifikasi (Acceptance Gates)

1. **Unit Tests Backend Rust (`cargo test`):**
   - Pengujian pemotongan petak `2x2` (`overlap 8%`), proyeksi balik koordinat lokal ke global, dan deduplikasi *Adaptive Radius Point-NMS*.
   - Pengujian parser respons `gemini-3.8-flash-high` (`[y, x]` SSE) dan `deepseek-v4.1-flash-1017` (`[x, y]` JSON + `data: [DONE]`).
   - Pengujian autentikasi passcode `2limanov` (menolak request tanpa sesi dengan `401`, menerima setelah `/api/auth/unlock`).
2. **Frontend Build & Typecheck:**
   - `npm run build` di `prototype/web/` menghasilkan bundle produksi bersih tanpa error TypeScript (`tsc -b`).
3. **Verifikasi End-to-End Live di `https://testing.rkhyg.xyz`:**
   - Cek sertifikat TLS & HTTP 200 pada `https://testing.rkhyg.xyz`.
   - Cek `GET /api/samples` dan `GET /api/packs` mengembalikan 11 sampel riset dan pack `apple` (`2.89 MB`).
   - Uji `POST /api/auth/unlock` dengan `2limanov`, unggah sampel gambar, jalankan Compare Mode (`M2` + `M3` + `M6`) dan uji `M1` untuk memastikan `Semaphore(1)` dan streaming SSE mengembalikan koordinat titik serta gambar hasil render dengan benar.
   - Uji tampilan visual desktop & mobile serta fungsi klik tambah/hapus titik pada `InteractiveCanvas` dan tombol *Theme Toggle* (Dark <-> Light Cream).
