# Product Requirements Document: GambleRisk Guard

| | |
|:---|:---|
| **Versi** | 1.0 (draf) |
| **Status** | In development |
| **Tanggal** | 18 November 2026 |

---

## 1. Ringkasan

GambleRisk Guard adalah aplikasi web untuk **profiling risiko perilaku taruhan daring**. Aplikasi ini mengelompokkan akun berdasarkan cara mereka bertaruh, yaitu frekuensi, nominal, win rate, dan target pengali pencairan. Setiap kelompok kemudian dipetakan ke level risiko beserta rekomendasi tindakan yang bisa ditinjau oleh analis.

Inti produk adalah model K-Means ($k=5$) yang sudah dievaluasi. Model ini dilengkapi Isolation Forest untuk menandai anomali dan Gaussian Mixture Model (GMM) untuk skor keyakinan.

## 2. Latar Belakang & Masalah

* Pola taruhan berisiko, misalnya target pengali tinggi dengan win rate rendah, sulit dikenali secara manual dari ribuan log transaksi.
* Analis membutuhkan cara cepat untuk (1) memeriksa satu akun, (2) memindai banyak akun sekaligus, dan (3) memahami peta segmen secara visual.
* Model hasil riset biasanya berhenti di notebook dan tidak bisa langsung dipakai oleh pengguna non-teknis.

**Pernyataan masalah:** analis risiko belum punya alat yang mengubah log taruhan menjadi segmen perilaku dan level risiko yang mudah dipahami, dapat diaudit, dan transparan terhadap keterbatasan modelnya.

## 3. Tujuan & Non-Tujuan

### Tujuan
1. Menyediakan prediksi segmen dan level risiko untuk satu akun dalam hitungan detik.
2. Memproses file CSV berisi banyak akun dan menghasilkan file berlabel yang bisa diunduh.
3. Menyajikan visualisasi yang menjelaskan posisi akun terhadap segmen referensi.
4. Menjamin hasil aplikasi **identik** dengan model yang sudah dievaluasi (tanpa drift saat deployment).
5. Menampilkan keterbatasan model secara jujur di dalam aplikasi.

### Non-Tujuan
* Mendiagnosis atau memvonis individu sebagai penjudi bermasalah.
* Mengambil tindakan otomatis (blokir akun, pembatasan transaksi) tanpa review manusia.
* Memproses data mutasi bank/e-wallet. Fitur ini membutuhkan pelatihan ulang dan berada di luar cakupan versi ini.
* Autentikasi pengguna, penyimpanan data permanen, atau integrasi API eksternal.

## 4. Pengguna

| Persona | Kebutuhan | Modul Utama |
|:---|:---|:---|
| **Analis risiko / responsible gambling** | Memeriksa akun tertentu dan memutuskan perlu tindak lanjut atau tidak | Single Profile Simulator |
| **Tim kepatuhan / audit** | Memindai banyak akun secara berkala dan menyimpan hasil untuk laporan | Batch CSV Auditor |
| **Manajer / pemangku kepentingan** | Memahami gambaran besar sebaran risiko dan profil segmen | Analytics Dashboard |

## 5. User Stories

| ID | Sebagai | Saya ingin | Agar |
|:---|:---|:---|:---|
| US-01 | Analis | memasukkan metrik satu akun dan melihat level risikonya | bisa memutuskan tindak lanjut dengan cepat |
| US-02 | Analis | melihat seberapa yakin model terhadap hasilnya | tidak salah ambil keputusan untuk akun di perbatasan segmen |
| US-03 | Analis | membandingkan profil akun dengan rata-rata segmen | memahami alasan akun masuk segmen tertentu |
| US-04 | Tim kepatuhan | mengunggah CSV berisi ratusan akun | tidak perlu memeriksa satu per satu |
| US-05 | Tim kepatuhan | mengunggah log taruhan mentah | tidak perlu mengagregasi data sendiri |
| US-06 | Tim kepatuhan | mengunduh hasil berlabel sebagai CSV | bisa dilampirkan di laporan audit |
| US-07 | Tim kepatuhan | mengetahui baris yang tidak valid dan alasannya | data tidak hilang diam-diam |
| US-08 | Manajer | melihat proporsi akun per level risiko dan estimasi perputaran dana | memahami skala eksposur |
| US-09 | Semua | melihat peringatan bila input di luar rentang data latih | tidak terlalu percaya pada hasil ekstrapolasi |

## 6. Kebutuhan Fungsional

### 6.1 Pipeline Inferensi (Inti)

| ID | Kebutuhan | Kriteria Penerimaan |
|:---|:---|:---|
| FR-01 | Transformasi fitur identik dengan pelatihan: log1p pada `total_bets`, `avg_bet`, `avg_cashout`; `win_rate` apa adanya; lalu `scaler.transform` | Uji konsistensi: label 1.111 profil referensi 100% sama dengan notebook pelatihan |
| FR-02 | Prediksi segmen dengan K-Means; ID klaster dipetakan lewat `models/segment_map.json` | Tidak ada logika yang bergantung pada urutan indeks klaster |
| FR-03 | Pemetaan segmen → level risiko sesuai tabel Bagian 7 | Setiap segmen menghasilkan level yang benar pada data uji |
| FR-04 | Skor keyakinan dari GMM `predict_proba`; status **perbatasan** bila probabilitas maksimum < 0,6 | Skor tampil di Modul 1 dan 2 |
| FR-05 | Isolation Forest menandai akun anomali sebagai "Review manual"; akun tetap diberi segmen terdekat sebagai informasi | Akun anomali tidak pernah dibuang dari hasil |
| FR-06 | Rekomendasi tindakan per level risiko, bersifat saran | Teks rekomendasi muncul untuk setiap level |

### 6.2 Modul 1: Single Profile Simulator

| ID | Kebutuhan | Kriteria Penerimaan |
|:---|:---|:---|
| FR-10 | Input 4 fitur via slider/number input dengan rentang default P1–P99 data latih (dari `models/reference_stats.json`) | Nilai default dan rentang sesuai statistik referensi |
| FR-11 | Peringatan ekstrapolasi bila nilai di luar P1–P99 | Peringatan muncul tanpa memblokir prediksi |
| FR-12 | Output: badge level risiko, nama segmen, skor keyakinan, status perbatasan/anomali | Semua elemen tampil setelah submit |
| FR-13 | Radar chart: akun vs median segmen vs median populasi | Skala sumbu dinormalisasi dan konsisten |
| FR-14 | Disclaimer bahwa hasil adalah indikasi, bukan diagnosis | Selalu terlihat di halaman hasil |

### 6.3 Modul 2: Batch CSV Auditor

| ID | Kebutuhan | Kriteria Penerimaan |
|:---|:---|:---|
| FR-20 | Menerima CSV teragregasi: `account_id, total_bets, avg_bet, win_rate, avg_cashout` (opsional `total_profit`, tidak masuk model) | File contoh dari `data/sample_test.csv` terproses tanpa error |
| FR-21 | Menerima CSV log mentah: `account_id, bet, cashed_out` (kosong = kalah); agregasi identik dengan notebook (termasuk `avg_cashout` = 1,0 bila tidak pernah menang) | Hasil agregasi sama dengan notebook pada sampel uji |
| FR-22 | Akun dengan < 10 taruhan diberi status "Data belum cukup" | Tidak diprediksi, tetapi tetap muncul di hasil |
| FR-23 | Validasi: kolom wajib, tipe numerik, `win_rate` ∈ [0, 1], nilai non-negatif, file kosong | Baris tidak valid ditampilkan terpisah beserta alasannya; aplikasi tidak crash |
| FR-24 | Tabel hasil dengan filter per level risiko | Filter memperbarui tabel secara langsung |
| FR-25 | Unduh hasil sebagai CSV (kolom asli + segmen, level risiko, skor keyakinan, flag anomali, rekomendasi) | File terunduh dan bisa dibuka di spreadsheet |

### 6.4 Modul 3: Analytics Dashboard

| ID | Kebutuhan | Kriteria Penerimaan |
|:---|:---|:---|
| FR-30 | Scatter PCA 2D dan 3D berisi profil referensi per segmen, plus akun dari Modul 1/2 bila ada | Explained variance ratio tampil di label sumbu, misalnya "PC1 (xx,x%)" |
| FR-31 | Distribusi jumlah akun per level risiko | Total sama dengan jumlah akun yang diprediksi |
| FR-32 | Estimasi perputaran dana per level risiko (Σ `total_bets × avg_bet`) | Angka konsisten dengan tabel Modul 2 |
| FR-33 | Perbandingan profil median antar segmen | Nilai sesuai `reference_stats.json` |

## 7. Spesifikasi Model & Data

**Data latih:** 1.111 profil pemain (minimal 10 taruhan) dari 50.000 taruhan platform crash game Bustabit.

**Fitur:** `total_bets`, `avg_bet` (bits), `win_rate`, `avg_cashout`. `total_profit` dikeluarkan dari fitur karena merupakan variabel hasil.

**Model:** K-Means $k=5$ (`random_state=42`, `n_init=10`). Silhouette 0,2588; DBI 1,1430; ARI antar seed ≥ 0,99.

| Level Risiko | Segmen | n | win_rate | avg_cashout | % rugi | ROI |
|:---|:---|---:|---:|---:|---:|---:|
| 🟢 Rendah | Target Rendah, Win Rate Tinggi | 297 | 0,80 | 1,19x | 40% | +4,4% |
| 🟡 Perlu Dipantau | Frekuensi Tinggi, Target Rendah | 235 | 0,68 | 1,37x | 51% | +4,4% |
| 🟡 Perlu Dipantau | Taruhan Nominal Tinggi | 226 | 0,60 | 1,49x | 43% | +7,1% |
| 🔴 Tinggi | Target Menengah | 244 | 0,33 | 2,30x | 61% | −24,6% |
| 🔴 Tinggi | Target Cashout Tinggi | 109 | 0,14 | 9,05x | 61% | −23,3% |
| ⚪ Review manual | Anomali (Isolation Forest) | – | – | – | – | – |

Segmen "Taruhan Nominal Tinggi" menguasai 88,6% perputaran dana, sehingga dipantau dari sisi eksposur, bukan kerugian.

**Artefak di `models/`:** `scaler.pkl`, `kmeans.pkl`, `gmm.pkl`, `isolation_forest.pkl`, `segment_map.json`, `reference_stats.json`.

## 8. Kebutuhan Non-Fungsional

| ID | Kategori | Kebutuhan |
|:---|:---|:---|
| NFR-01 | Performa | Prediksi satu akun < 1 detik; CSV 10.000 baris < 10 detik |
| NFR-02 | Reproduktibilitas | Versi `scikit-learn` di-pin sama persis dengan versi saat artefak diekspor |
| NFR-03 | Ketersediaan | Ter-deploy di Streamlit Community Cloud dan dapat diakses publik |
| NFR-04 | Privasi | Data yang diunggah hanya diproses di memori sesi; tidak disimpan |
| NFR-05 | Transparansi | Disclaimer, skor keyakinan, dan peringatan ekstrapolasi selalu ditampilkan |
| NFR-06 | Usability | Bisa dipakai tanpa membaca dokumentasi; setiap error memberi pesan yang jelas |
| NFR-07 | Maintainability | Logika model di `src/`, UI di `pages/`; tidak ada logika model di file UI |

## 9. Metrik Keberhasilan

| Metrik | Target |
|:---|:---|
| Kesesuaian label dengan notebook pelatihan | 100% |
| Cakupan skenario demo (setiap level risiko + anomali + input tidak valid) | 100% |
| Tes otomatis lulus (`pytest`) | 100% |
| Aplikasi dapat diakses dari perangkat lain saat demo | Ya |
| Error tak tertangani selama uji end-to-end | 0 |

## 10. Rencana Rilis

| Milestone | Isi | Definisi Selesai |
|:---|:---|:---|
| **M1: Fondasi** | Artefak + metadata diekspor, skema CSV disepakati, kerangka aplikasi berjalan | `streamlit run app.py` berjalan; artefak ada di `models/` |
| **M2: Inti** | Pipeline inferensi, GMM, Isolation Forest, Modul 1 | Uji konsistensi lulus; Modul 1 berfungsi penuh |
| **M3: Fitur lengkap** | Modul 2 & 3, data demo, deploy pertama | Semua FR terpenuhi; URL publik aktif |
| **M4: Rilis** | Uji end-to-end, perbaikan bug, dokumentasi final | Semua metrik Bagian 9 tercapai |

## 11. Risiko & Mitigasi

| Risiko | Dampak | Mitigasi |
|:---|:---|:---|
| Versi `scikit-learn` berbeda membuat pickle gagal dimuat | Aplikasi tidak jalan di cloud | Pin versi; uji deploy sejak M3 |
| ID klaster tertukar saat ekspor ulang | Level risiko salah | `segment_map.json` diturunkan dari centroid + uji pemetaan |
| Pengguna memasukkan data di luar distribusi latih | Hasil tidak bermakna | Peringatan ekstrapolasi + Isolation Forest |
| Hasil dibaca sebagai vonis | Keputusan yang tidak adil | Disclaimer, bahasa "indikasi", rekomendasi bersifat saran |
| Struktur klaster lemah-sedang (Silhouette ≈ 0,26) | Akun perbatasan rawan salah segmen | Skor keyakinan GMM + status perbatasan |

## 12. Pengembangan Lanjutan

* Pelatihan ulang pada data mutasi rekening/e-wallet berlabel untuk kebutuhan anti-money laundering.
* Pemantauan berkala dengan riwayat perubahan segmen per akun.
* API (FastAPI) agar bisa diintegrasikan ke sistem lain.

## 13. Pertanyaan Terbuka

* Berapa nilai `contamination` Isolation Forest yang dipakai? Ditentukan setelah benchmark.
* Apakah ambang perbatasan 0,6 sudah tepat? Dievaluasi dari distribusi `predict_proba` pada data latih.
* Versi `scikit-learn` yang dipakai saat ekspor artefak di Colab.
