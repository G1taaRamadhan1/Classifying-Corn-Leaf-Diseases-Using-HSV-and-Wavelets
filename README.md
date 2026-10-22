# Klasifikasi Penyakit Daun Jagung dengan Machine Learning dan Fitur Wavelet

Repositori ini berisi notebook eksperimen untuk membandingkan K-Nearest Neighbors (KNN), Support Vector Machine (SVM), dan Random Forest dalam klasifikasi citra daun jagung. Eksperimen membandingkan fitur dasar dari kanal Value (V) pada ruang warna HSV dengan fitur gabungan kanal V dan statistik koefisien wavelet.

## Tujuan

Eksperimen ini berfokus pada tiga hal:

- Membandingkan performa KNN, SVM, dan Random Forest menggunakan fitur dasar kanal V HSV.
- Menguji apakah penambahan fitur wavelet Haar, Daubechies 4 (db4), dan Coiflet 1 (coif1) pada level dekomposisi 1, 2, dan 3 mengubah performa klasifikasi.
- Membandingkan waktu validasi silang, pelatihan model, dan prediksi pada data pengujian berdasarkan pengukuran yang tersedia di notebook.

## Dataset

Dataset yang digunakan adalah [Corn Leaf Disease di Kaggle](https://www.kaggle.com/datasets/ndisan/corn-leaf-disease/code), diunduh melalui `kagglehub`.

Empat kelas yang digunakan:

- Bercak Daun
- Daun Sehat
- Hawar Daun
- Karat Daun

Pemeriksaan awal notebook menemukan 4.000 berkas gambar JPG, masing-masing 1.000 gambar per kelas. Pemeriksaan hash SHA-256 menemukan enam salinan gambar identik tambahan. Setelah deduplikasi, eksperimen menggunakan 3.994 gambar.

| Tahap | Jumlah gambar |
|---|---:|
| Sebelum deduplikasi | 4.000 |
| Salinan identik tambahan yang dikeluarkan | 6 |
| Setelah deduplikasi | 3.994 |
| Data pelatihan (80%) | 3.195 |
| Data pengujian (20%) | 799 |

Pembagian data menggunakan `train_test_split` dengan `stratify` dan `random_state=42`. Dataset tidak disertakan di repositori ini; notebook mengunduhnya saat dijalankan.

## Alur eksperimen

1. Mengunduh dan memeriksa dataset serta resolusi gambar.
2. Memeriksa file gambar yang dapat dibaca dan mencari duplikasi identik menggunakan hash SHA-256.
3. Menghapus salinan identik tambahan agar satu gambar tidak muncul berulang dalam eksperimen.
4. Mengubah ukuran gambar menjadi 256 × 256 dengan mempertahankan rasio aspek dan menambahkan padding putih.
5. Mengonversi gambar ke HSV dan mengambil kanal V.
6. Membentuk fitur baseline dari kanal V yang diubah ukurannya menjadi 64 × 64, lalu diratakan menjadi vektor 4.096 nilai.
7. Mengekstrak statistik subband dari dekomposisi wavelet 2D untuk Haar, db4, dan coif1 pada level 1, 2, dan 3.
8. Menggabungkan vektor fitur baseline V dengan fitur statistik wavelet.
9. Melatih dan mengevaluasi KNN, SVM, dan Random Forest.
10. Membandingkan akurasi, precision, recall, F1-score, confusion matrix, serta waktu komputasi yang dicatat.

Notebook juga memuat pemeriksaan awal segmentasi GrabCut. Segmentasi tersebut merupakan bagian eksplorasi; jalur ekstraksi fitur yang digunakan pada perbandingan model mengambil kanal V dari gambar yang telah diubah ukurannya, bukan hasil segmentasi GrabCut.

## Ekstraksi fitur wavelet

Notebook menerapkan dekomposisi wavelet diskrit dua dimensi (`pywt.wavedec2`) pada kanal V yang dinormalisasi ke rentang 0–1. Statistik dihitung untuk subband aproksimasi pada level tertinggi serta subband detail horizontal, vertikal, dan diagonal pada setiap level.

Empat statistik yang dihitung untuk setiap subband:

- Rata-rata
- Simpangan baku
- Rata-rata nilai absolut
- Akar dari rata-rata kuadrat (RMS)

Dengan implementasi ini, fitur wavelet menghasilkan 16 nilai pada level 1, 28 nilai pada level 2, dan 40 nilai pada level 3. Fitur tersebut digabungkan dengan 4.096 nilai baseline V, sehingga setiap vektor fitur gabungan berisi 4.112, 4.124, atau 4.136 nilai sesuai level dekomposisi.

## Model dan konfigurasi

| Model | Konfigurasi dalam notebook |
|---|---|
| KNN | `n_neighbors=5`, `weights="uniform"`, `n_jobs=-1`, dengan `StandardScaler` |
| SVM | `kernel="rbf"`, `C=1.0`, `gamma="scale"`, dengan `StandardScaler` |
| Random Forest | `n_estimators=200`, `random_state=42`, `n_jobs=-1` |

Validasi silang menggunakan `StratifiedKFold` dengan 5 fold, `shuffle=True`, dan `random_state=42`. Validasi silang dijalankan pada data pelatihan; data pengujian dipakai untuk evaluasi akhir pada pembagian data tersebut.

## Ringkasan hasil yang tersimpan di notebook

Angka berikut berasal dari output notebook yang tersedia, bukan dari eksperimen ulang saat repositori ini dibuat.

### Baseline: kanal V HSV

| Model | Akurasi CV (rata-rata) | Akurasi uji | Precision macro | Recall macro | F1 macro |
|---|---:|---:|---:|---:|---:|
| KNN | 73,87% | 73,22% | 75,80% | 73,17% | 72,32% |
| SVM | 79,56% | 80,98% | 81,56% | 80,96% | 81,20% |
| Random Forest | 82,25% | 83,10% | 83,35% | 83,09% | 83,20% |

### Konfigurasi wavelet dengan akurasi uji tertinggi pada setiap model

| Model | Fitur | Akurasi CV (rata-rata) | Akurasi uji | F1 macro |
|---|---|---:|---:|---:|
| KNN | Haar level 3 | 74,24% | 73,97% | 73,12% |
| SVM | db4 level 3 | 81,47% | 83,48% | 83,66% |
| Random Forest | coif1 level 3 | 89,55% | 89,99% | 90,01% |

Ada perbedaan antara konfigurasi terbaik menurut rata-rata cross-validation dan konfigurasi dengan akurasi tertinggi pada data uji. Untuk Random Forest, db4 level 3 memiliki rata-rata CV tertinggi (89,83%), sedangkan coif1 level 3 memiliki akurasi uji tertinggi (89,99%). Keduanya tidak boleh dianggap sebagai ukuran yang saling menggantikan.

Random Forest mencapai akurasi training 100% pada eksperimen ini. Hal tersebut merupakan tanda potensi overfitting dan perlu diperhatikan ketika menafsirkan hasil.

### Waktu komputasi

Notebook mencatat waktu ekstraksi seluruh representasi fitur sebesar **618,07 detik** (sekitar 10 menit 18 detik). Ini merupakan waktu total untuk proses ekstraksi fitur seluruh gambar dan seluruh konfigurasi, bukan waktu tambahan khusus untuk satu model.

Waktu validasi silang, pelatihan model, dan prediksi pengujian juga dicatat secara terpisah. Durasi dapat berubah sesuai perangkat keras, versi pustaka, dan kondisi lingkungan saat notebook dijalankan.

## Menjalankan notebook

Disarankan menggunakan Google Colab atau lingkungan Python dengan akses internet untuk mengunduh dataset.

1. Buka notebook `comparisson_ml_wavelet(1).ipynb` di Google Colab atau Jupyter Notebook.
2. Pastikan koneksi internet tersedia.
3. Jalankan sel secara berurutan dari atas ke bawah.
4. Tunggu proses unduh dataset, audit gambar, ekstraksi fitur, validasi silang, dan evaluasi selesai.

Notebook memasang beberapa dependensi pada sel awal. Jika ingin memasang dependensi secara manual, pustaka utama yang digunakan adalah:

```bash
pip install kagglehub PyWavelets opencv-python-headless numpy pandas matplotlib scikit-learn pillow
```

Eksperimen memerlukan waktu karena melakukan hashing file, ekstraksi fitur dari seluruh gambar, dan validasi silang untuk beberapa kombinasi model dan wavelet. Output yang tersimpan di notebook dapat dibaca tanpa menjalankan ulang seluruh eksperimen, tetapi angka tersebut tetap perlu dipahami sebagai hasil dari eksekusi sebelumnya.

## Struktur repositori

Struktur sederhana yang disarankan:

```text
.
├── README.md
└── comparisson_ml_wavelet(1).ipynb
```

Jika nama notebook ingin dirapikan sebelum diunggah, nama seperti `klasifikasi_daun_jagung_wavelet.ipynb` dapat digunakan. Sesuaikan nama pada README dengan nama file yang benar-benar ada di repositori.

## Catatan dan keterbatasan

- Distribusi jumlah gambar per kelas relatif seimbang setelah deduplikasi, tetapi karakteristik resolusi dan sumber gambar tidak seragam. Kelas Bercak Daun memiliki pola resolusi yang berbeda dari kelas lainnya, sehingga model mungkin memanfaatkan karakteristik sumber gambar selain ciri penyakit.
- Hasil berasal dari satu pembagian train-test dengan `random_state=42`. Validasi silang dilakukan pada data pelatihan, tetapi belum menggantikan pengujian pada dataset eksternal.
- Random Forest memperoleh akurasi training 100%; pemeriksaan generalisasi lebih lanjut disarankan.
- Waktu komputasi yang tercatat bergantung pada lingkungan eksekusi dan tidak dapat dianggap sebagai tolok ukur perangkat keras yang universal.
- Notebook tidak mencatat secara lengkap seluruh informasi perangkat keras dan versi semua pustaka. Untuk reproduksibilitas yang lebih baik, simpan versi dependensi dan spesifikasi lingkungan ketika eksperimen dijalankan kembali.
- Hasil ini merupakan eksperimen klasifikasi citra dan bukan sistem diagnosis pertanian siap pakai.

## Sumber

- Dataset: [Corn Leaf Disease — Kaggle](https://www.kaggle.com/datasets/ndisan/corn-leaf-disease/code)
- Pustaka wavelet: [PyWavelets](https://pywavelets.readthedocs.io/)
- Machine learning: [scikit-learn](https://scikit-learn.org/stable/)
- Pengolahan citra: [OpenCV](https://opencv.org/)

