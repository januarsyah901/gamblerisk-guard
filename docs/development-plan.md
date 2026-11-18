# Rancangan Pengembangan Proyek UAS: GambleRisk Guard
**Mata Kuliah:** Praktikum Penambangan Data (SVPL261503)  
**Program Studi:** Sarjana Terapan (D.4) Teknologi Rekayasa Perangkat Lunak, Sekolah Vokasi UGM  
**Dosen Pengampu:** Dr.Eng. Ir. Ganjar Alfian, S.T., M.Eng. & Dr. Imam Fahrurrozi, S.T., M.Cs.  
**Kelompok:** 4

> Dokumen ini adalah rencana kerja tim. Kebutuhan produk yang mengikat (ID fitur, kriteria penerimaan, kebutuhan non-fungsional) ada di [`PRD.md`](PRD.md). Bila ada perbedaan, PRD yang dipakai. Nama repositori, aplikasi, dan README memakai penamaan produk tanpa unsur mata kuliah (lihat Bagian 8).

---

## 1. Latar Belakang & Transisi Fase (UTS ke UAS)

* **Fase UTS (25%):** selesai sampai tahap *Model Evaluation & Interpretation*. Dataset Bustabit (50.000 taruhan) diagregasi menjadi profil per pemain, disaring ke pemain stabil (minimal 10 taruhan), lalu dilatih K-Means final dengan $k=5$.
* **Fase UAS (25%):** membawa model yang sudah dievaluasi ke tahap *Model Deployment* dalam bentuk aplikasi web interaktif berbasis **Streamlit**.

**Posisi produk:** GambleRisk Guard adalah alat bantu **profiling perilaku taruhan** untuk analis risiko (operator platform, regulator, atau tim *responsible gambling*). Aplikasi ini mengelompokkan pola perilaku, **bukan** memvonis seseorang sebagai penjudi bermasalah. Setiap output adalah indikasi yang tetap memerlukan review manusia (*human-in-the-loop*).

---

## 2. Fondasi Baseline dari Model UTS

Seluruh pengembangan mengacu pada notebook `PPD_UTS_Kelompok_4_KMeans_Clustering_FINAL.ipynb`.

* **Cakupan data:** 1.111 dari 4.149 pemain (26,8%) dengan `total_bets >= 10`.
* **Fitur clustering (murni perilaku):**
  1. `total_bets`: frekuensi bermain (log1p).
  2. `avg_bet`: rata-rata nominal taruhan dalam *bits* (log1p).
  3. `win_rate`: rasio kemenangan (0 sampai 1, tanpa transformasi).
  4. `avg_cashout`: rata-rata target pengali saat menang (log1p; diisi 1,0 bila belum pernah menang).

  `total_profit` **tidak** dipakai sebagai fitur karena merupakan variabel hasil (*outcome*), bukan perilaku. Variabel ini hanya dipakai untuk interpretasi klaster.
* **Konfigurasi model:** K-Means $k=5$, `random_state=42`, `n_init=10`. $k=5$ dipilih sebagai kompromi: Silhouette 0,2588 dan DBI 1,1430. Struktur klaster tergolong lemah-sedang, sehingga klaster dibaca sebagai segmen deskriptif, bukan kategori tegas.
* **Stabilitas:** ARI antar 5 seed ≥ 0,99; ARI subsample 80% rata-rata 0,939.
* **Artefak:** `scaler.pkl` (StandardScaler) dan `kmeans.pkl` (K-Means final).

### 2.1 Profil 5 Segmen (median, data UTS)

| Segmen | n | total_bets | avg_bet | win_rate | avg_cashout | % pemain rugi | ROI | Porsi volume |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| Target Rendah, Win Rate Tinggi | 297 | 17 | 43 | 0,80 | 1,19x | 40% | +4,4% | 0,6% |
| Frekuensi Tinggi, Target Rendah | 235 | 69 | 39 | 0,68 | 1,37x | 51% | +4,4% | 6,0% |
| Taruhan Nominal Tinggi | 226 | 20 | 2.225 | 0,60 | 1,49x | 43% | +7,1% | 88,6% |
| Target Menengah | 244 | 16 | 84 | 0,33 | 2,30x | 61% | −24,6% | 1,0% |
| Target Cashout Tinggi | 109 | 36 | 106 | 0,14 | 9,05x | 61% | −23,3% | 3,8% |

Catatan interpretasi: hubungan antara target pengali, win rate, dan kerugian bersifat **asosiasi**, bukan kausal. Label segmen adalah label deskriptif buatan analis. Istilah seperti "chasing loss" atau "bot" tidak dipakai sebagai klaim karena data tidak memiliki label pembanding.

---

## 3. Pemetaan Segmen ke Level Risiko

ID klaster di `kmeans.pkl` (0 sampai 4) bersifat acak. Karena itu, pemetaan ke nama segmen **dikunci berdasarkan karakteristik centroid**, mengikuti aturan yang sama dengan notebook UTS: `avg_bet` tertinggi → Nominal Tinggi; dari sisanya, `avg_cashout` tertinggi → Cashout Tinggi; lalu `total_bets` tertinggi → Frekuensi Tinggi; sisa dua klaster diurutkan berdasarkan `avg_cashout`. Hasil pemetaan disimpan sebagai `models/segment_map.json`, sehingga aplikasi tidak bergantung pada urutan indeks.

| Level Risiko | Segmen | Dasar Penentuan |
|:---|:---|:---|
| 🟢 **Rendah** | Target Rendah, Win Rate Tinggi | ROI positif, win rate tertinggi, proporsi pemain rugi terendah (40%) |
| 🟡 **Perlu Dipantau** | Frekuensi Tinggi, Target Rendah | Intensitas bermain tertinggi (median 69 taruhan); risiko dari sisi frekuensi, bukan kerugian |
| 🟡 **Perlu Dipantau** | Taruhan Nominal Tinggi | Menguasai 88,6% perputaran dana; risiko dari sisi eksposur nominal, bukan kerugian |
| 🔴 **Tinggi** | Target Menengah | ROI −24,6%, 61% pemain rugi |
| 🔴 **Tinggi** | Target Cashout Tinggi | ROI −23,3%, win rate 0,14, 61% pemain rugi |
| ⚪ **Review manual** | Anomali (di luar segmen) | Ditandai Isolation Forest (Bagian 6). Input di luar rentang data latih **tidak** otomatis masuk level ini; hanya memunculkan peringatan ekstrapolasi |

Setiap level disertai **rekomendasi tindakan yang bersifat saran**, misalnya: tanpa tindakan / pemantauan berkala / pesan *responsible gambling* dan opsi batas setor mandiri / eskalasi ke analis.

---

## 4. Batas Lingkup Domain & Arah Pengembangan Finansial

**Keputusan desain:** aplikasi menerima input dalam **ruang fitur perilaku taruhan** yang sama dengan data latih (frekuensi, nominal dalam *bits*, win rate, target pengali). Alasannya:

1. `scaler.pkl` dilatih pada distribusi Bustabit. Memasukkan Rupiah atau rasio mutasi bank membuat nilai terstandardisasi jauh di luar rentang data latih, sehingga K-Means hanya memilih centroid terdekat tanpa makna perilaku.
2. Variabel mutasi bank tidak setara secara makna. Contohnya, *Deposit-to-Withdrawal Ratio* yang tinggi lebih mirip pemain yang menang, sedangkan `avg_cashout` yang tinggi justru berasosiasi dengan kerugian.

Pemetaan ke mutasi rekening/e-wallet diposisikan sebagai **kerja lanjutan** dan dibahas di laporan (bab Batasan & Pengembangan), bukan fitur aplikasi:

| Fitur Perilaku (Model) | Kandidat Padanan Finansial | Syarat Agar Valid |
|:---|:---|:---|
| `total_bets` | Frekuensi transfer ke merchant/rekening tujuan terindikasi | Rekening tujuan harus teridentifikasi |
| `avg_bet` | Rata-rata nominal transfer keluar ke tujuan tersebut | Dilatih ulang dalam satuan Rupiah |
| `win_rate` | Rasio transaksi masuk dari tujuan yang sama terhadap transaksi keluar | Butuh pasangan transaksi keluar-masuk |
| `avg_cashout` | Rasio nominal masuk terhadap nominal keluar per siklus | Arah risiko harus divalidasi ulang |

Kesimpulan untuk laporan: penerapan di perbankan membutuhkan **pelatihan ulang** model pada data mutasi berlabel, bukan sekadar memetakan nama kolom.

---

## 5. Arsitektur Aplikasi (Streamlit)

```text
Input (form / CSV)
  → Validasi skema; total_bets < 10 → "Data belum cukup" (tidak diprediksi)
  → Cek rentang P1–P99 data latih → peringatan ekstrapolasi (tidak memblokir)
  → log1p (total_bets, avg_bet, avg_cashout)
  → scaler.transform
  → Isolation Forest   ── anomali ──→ level "Review manual" (tetap diberi segmen terdekat sebagai info)
  → kmeans.predict     → segment_map.json → level risiko + rekomendasi
  → GMM predict_proba  → skor keyakinan & penanda "zona perbatasan"
```

### Modul 1: Single Profile Simulator
* **Input:** slider/number input untuk 4 fitur. Rentang default diambil dari persentil 1 sampai 99 data latih. Input di luar rentang tetap boleh, tetapi memunculkan peringatan ekstrapolasi.
* **Output:**
  * Badge level risiko dan nama segmen.
  * Skor keyakinan dari GMM. Bila probabilitas tertinggi < 0,6, akun ditandai **zona perbatasan**.
  * Radar chart: profil input dibandingkan median segmen dan median populasi.
  * Rekomendasi tindakan beserta disclaimer bahwa hasil bukan diagnosis.

### Modul 2: Batch CSV Auditor
* **Skema CSV wajib (data sudah teragregasi per akun):** `account_id, total_bets, avg_bet, win_rate, avg_cashout`. Kolom opsional: `total_profit` (hanya untuk ringkasan, tidak masuk model).
* **Opsi input transaksi mentah:** CSV per taruhan dengan kolom `account_id, bet, cashed_out` (kosong bila kalah). Aplikasi melakukan agregasi yang **identik** dengan notebook UTS, termasuk aturan `avg_cashout` = 1,0 untuk akun yang belum pernah menang.
* **Minimal 10 taruhan (kedua format):** akun dengan `total_bets` < 10 diberi status "Data belum cukup", tidak diprediksi, tetapi tetap muncul di hasil.
* **Validasi:** cek kolom wajib, tipe numerik, `win_rate` di [0, 1], nilai non-negatif, file kosong; baris yang tidak valid ditampilkan terpisah beserta alasannya, bukan dibuang diam-diam.
* **Output:** tabel hasil (kolom asli + segmen, level risiko, skor keyakinan, flag anomali, rekomendasi), filter per level, dan tombol unduh CSV berlabel.

### Modul 3: Analytics Dashboard
* Scatter plot PCA 2D/3D (Plotly) berisi data latih per segmen ditambah titik akun dari Modul 1/2. **Explained variance ratio** ditampilkan di label sumbu (misalnya "PC1 (xx,x%)").
* Distribusi proporsi akun per level risiko.
* Estimasi total perputaran dana (Σ `total_bets × avg_bet`) per level risiko.
* Grafik perbandingan profil median antar segmen.

---

## 6. Penyempurnaan Teknis untuk UAS

1. **Isolation Forest sebagai penanda anomali, bukan penyaring.**
   * Dilatih pada data latih yang sudah distandardisasi; `contamination` dipilih dan dilaporkan (misalnya 0,03 sampai 0,05).
   * Akun anomali **tidak dibuang**. Akun ekstrem justru paling relevan bagi analis, sehingga diberi label "Anomali, review manual".
   * K-Means **tidak dilatih ulang**, agar segmen tetap identik dengan hasil UTS yang sudah dievaluasi.
2. **GMM sebagai skor keyakinan, bukan model pengganti.**
   * GMM (5 komponen, `covariance_type` dipilih via BIC) diinisialisasi dari centroid K-Means agar komponennya sejajar dengan segmen.
   * Label utama tetap K-Means. `predict_proba` GMM hanya dipakai sebagai skor keyakinan dan penanda zona perbatasan.
   * Laporan menyajikan perbandingan: BIC/AIC, Silhouette, dan tingkat kesepakatan label K-Means vs GMM (ARI).
3. **Transparansi PCA:** explained variance ratio per komponen dan kumulatif ditampilkan di aplikasi dan laporan.

---

## 7. Rencana Pengujian

| Jenis Uji | Isi | Kriteria Lulus |
|:---|:---|:---|
| Uji konsistensi model | Jalankan pipeline aplikasi terhadap 1.111 data latih | Label 100% sama dengan label notebook UTS |
| Uji pemetaan segmen | Cek `segment_map.json` terhadap median tiap segmen | Sesuai tabel Bagian 2.1 |
| Unit test preprocessing | log1p, pengisian `avg_cashout`, agregasi transaksi mentah | Hasil sama dengan notebook pada sampel uji |
| Uji validasi input | CSV kolom hilang, nilai negatif, `win_rate` > 1, file kosong | Pesan error jelas, aplikasi tidak crash |
| Uji skenario demo | `data/sample_test.csv` berisi akun contoh untuk setiap level risiko + anomali | Tiap level muncul minimal sekali |
| Uji deployment | Aplikasi di Streamlit Community Cloud dibuka dari perangkat lain | Model ter-load, ketiga modul berjalan |
| Uji performa | Prediksi satu akun dan CSV 10.000 baris | < 1 detik dan < 10 detik (PRD NFR-01) |

**Kompatibilitas artefak:** catat `sklearn.__version__` dari Colab saat model disimpan, lalu pin versi yang sama persis di `requirements.txt`. Pickle scikit-learn bisa gagal atau berperilaku berbeda antar versi.

---

## 8. Struktur Repositori

Nama repositori: **`gamblerisk-guard`** (tanpa unsur mata kuliah).

```text
gamblerisk-guard/
├── README.md                     # Deskripsi produk, cara install & menjalankan, link demo
├── requirements.txt              # Dependensi dengan versi ter-pin
├── docs/
│   ├── PRD.md                    # Product Requirements Document
│   └── development-plan.md       # Dokumen rencana kerja ini
├── app.py                        # Entry point Streamlit
├── pages/                        # Modul 1–3 sebagai halaman Streamlit
├── models/
│   ├── scaler.pkl
│   ├── kmeans.pkl
│   ├── gmm.pkl
│   ├── isolation_forest.pkl
│   ├── segment_map.json          # Pemetaan ID klaster → segmen → level risiko
│   └── reference_stats.json      # Persentil & median data latih (rentang input, radar chart)
├── src/
│   ├── preprocessing.py          # Agregasi, validasi, log1p, scaling
│   ├── inference.py              # Pipeline prediksi, level risiko, rekomendasi
│   └── visualization.py          # Grafik Plotly (PCA, radar, distribusi)
├── notebooks/
│   ├── 01_kmeans_baseline.ipynb  # Notebook model final (dari UTS)
│   └── 02_gmm_iforest.ipynb      # Pelatihan & evaluasi GMM dan Isolation Forest
├── data/
│   ├── reference_profiles.csv    # Profil 1.111 akun data latih (untuk scatter & uji konsistensi)
│   └── sample_test.csv           # Data demo
└── tests/
    └── test_pipeline.py          # Uji konsistensi & validasi input
```

Laporan dan slide disimpan di luar repositori kode (folder tugas kuliah / Google Drive), supaya repositori tetap bersih dan profesional.

---

## 9. Pembagian Peran

Peran dibagi berdasarkan lingkup kerja. Silakan tiap anggota memilih satu peran dan mengisi kolom **Penanggung Jawab**.

| Peran | Lingkup Tugas | Luaran | Penanggung Jawab |
|:---|:---|:---|:---|
| **A. ML Engineer & Pipeline Integrator** | 1. Ekspor ulang artefak model + `segment_map.json` + `reference_stats.json` dari notebook UTS.<br>2. Membangun `src/inference.py` dan `src/preprocessing.py` (agregasi, validasi, scoring, level risiko, rekomendasi).<br>3. Menulis uji konsistensi model di `tests/`. | `src/inference.py`, `src/preprocessing.py`, `models/`, `tests/` | |
| **B. Algorithm Benchmark Engineer** | 1. Melatih GMM (pemilihan kovarians via BIC, inisialisasi dari centroid K-Means).<br>2. Melatih Isolation Forest dan menentukan `contamination`.<br>3. Menyusun tabel perbandingan K-Means vs GMM (BIC/AIC, Silhouette, ARI) untuk laporan. | `notebooks/02_gmm_iforest.ipynb`, `models/gmm.pkl`, `models/isolation_forest.pkl` | |
| **C. Frontend & Deployment Engineer** | 1. Membangun `app.py` dan halaman Modul 1 & 2 (form, badge, upload, tabel, unduh CSV, pesan error).<br>2. Menulis disclaimer & teks rekomendasi di UI.<br>3. Deploy ke Streamlit Community Cloud dan menulis README. | `app.py`, `pages/`, `README.md`, URL demo | |
| **D. Visualization & Data Specialist** | 1. Membangun `src/visualization.py` (PCA 2D/3D + explained variance, radar, distribusi risiko, perbandingan segmen) dan halaman Modul 3.<br>2. Menyiapkan `data/sample_test.csv` (skenario setiap level risiko + anomali) dan `data/reference_profiles.csv`. | `src/visualization.py`, `pages/` Modul 3, `data/` | |

**Tugas bersama (semua anggota):**
* **Laporan akhir:** tiap peran menulis bab sesuai bagiannya (A: arsitektur & pipeline; B: benchmark algoritma; C: UI/UX & deployment; D: visualisasi & skenario uji). Satu orang ditunjuk sebagai editor akhir: ______.
* **Slide & demo:** tiap peran menyiapkan slide bagiannya. Satu orang ditunjuk sebagai penyusun akhir dan pembawa demo: ______.

---

## 10. Timeline

| Minggu | Milestone PRD | Target | Peran Utama |
|:---|:---|:---|:---|
| 1 | M1: Fondasi | ~~Repo dibuat~~ (selesai), artefak + `segment_map.json` + `reference_stats.json` diekspor, kerangka `app.py` berjalan | A, C |
| 2 | M2: Inti | Pipeline inferensi + uji konsistensi lulus; GMM & Isolation Forest selesai; Modul 1 berjalan | A, B, C |
| 3 | M3: Fitur lengkap | Modul 2 & 3 selesai, data demo siap, deploy pertama ke Streamlit Cloud | C, D |
| 4 | M4: Rilis | Uji end-to-end, perbaikan bug, laporan & slide final, gladi demo | Semua |

---

## 11. Checklist Deliverables Akhir UAS

| No | Komponen | Format | Keterangan |
|:---|:---|:---|:---|
| 1 | Aplikasi web live demo | URL publik | Streamlit Community Cloud; diuji dari perangkat lain sebelum presentasi |
| 2 | Kode sumber | Repositori Git `gamblerisk-guard` | Kode, model, tes, README instalasi |
| 3 | Slide presentasi (10 menit) | PDF / Google Slides | Masalah, metode, demo langsung, batasan & pengembangan |
| 4 | Laporan akhir | PDF | Pengembangan laporan UTS + bab arsitektur sistem, benchmark algoritma, UI/UX, pengujian, dan batasan domain |
