# Sistem Rekomendasi Pembukaan Catur

Sebuah aplikasi web interaktif yang memberikan rekomendasi pembukaan catur yang dipersonalisasi berdasarkan preferensi dan rating Anda. Sistem ini menggunakan tiga pendekatan machine learning: Content-Based Filtering, Collaborative Filtering, dan Hybrid Filtering.

![image](https://github.com/user-attachments/assets/7808d89e-2b6e-4e11-89cb-785e2ffcc983)


## 🎯 Fitur Utama

- **Content-Based Filtering**: Rekomendasi berdasarkan kemiripan langkah dengan pembukaan favorit Anda
- **Collaborative Filtering**: Rekomendasi berdasarkan preferensi pemain dengan rating serupa
- **Hybrid Filtering**: Gabungan kedua metode untuk hasil yang optimal
- **Analisis Pembukaan**: Visualisasi dan statistik pembukaan favorit Anda
- **Interface Interaktif**: Antarmuka web yang mudah digunakan dengan Streamlit

## 🛠️ Teknologi yang Digunakan

- **Python 3.12**
- **Streamlit** - Framework aplikasi web
- **TensorFlow** - Deep learning untuk collaborative filtering
- **Pandas & NumPy** - Manipulasi dan analisis data
- **Matplotlib & Seaborn** - Visualisasi data
- **Scikit-learn** - Machine learning utilities
- **Pickle** - Serialisasi model

## 📋 Persyaratan Sistem

### Perlengkapan Sebelum Memulai

1. **Python 3.12**
2. **Git** (untuk clone repository)
3. **Minimum 4GB RAM** (untuk menjalankan model TensorFlow)
4. **Koneksi internet** (untuk download dependencies)

### Dependencies

Install semua dependencies dengan menjalankan:

```bash
pip install streamlit pandas numpy matplotlib seaborn tensorflow scikit-learn
```

Atau gunakan file requirements.txt (jika tersedia):

```bash
pip install -r requirements.txt
```

## 📁 Struktur Proyek

```
chess-opening-recommender/
├── app.py                          # Aplikasi Streamlit utama
├── games.csv                       # Dataset permainan catur
├── models/                         # Folder model yang sudah dilatih
│   ├── content_based_model.pkl     # Model content-based filtering
│   ├── collaborative_data.pkl      # Data untuk collaborative filtering
│   ├── collaborative_model.keras   # Model neural network
│   ├── collaborative_model_data.pkl # Data tambahan model collaborative
│   └── hybrid_model.pkl            # Model hybrid filtering
├── training_notebook.ipynb         # Notebook untuk melatih model (opsional)
└── README.md                       # Dokumentasi proyek
```

## 🚀 Cara Menjalankan Aplikasi

### 1. Clone Repository

```bash
git clone <repository-url>
cd chess-opening-recommender
```
### 2. Buat dan Aktiffkan Virtual Environment
```bash
# Buat virtual environment
python -m venv venv

# Aktifkan venv
# Untuk Windows:
venv\Scripts\activate

# Untuk macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Persiapkan Data dan Model

Pastikan Anda memiliki:
- File `games.csv` dalam direktori utama
- Folder `models/` dengan semua file model yang diperlukan

### 5. Jalankan Aplikasi

```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser pada alamat `http://localhost:8501`

## 📊 Dataset

Aplikasi ini menggunakan dataset permainan catur yang berisi:
- **ID permainan**: Identifikasi unik setiap permainan
- **Data pemain**: Rating dan ID pemain putih dan hitam
- **Informasi pembukaan**: Nama pembukaan, langkah-langkah, dan kategori
- **Hasil permainan**: Pemenang (putih/hitam/seri)
- **Metadata**: Jumlah langkah pembukaan dan klasifikasi

## 🎮 Cara Menggunakan

### 1. Input Profil Anda
- Masukkan rating catur Anda (500-3000)
- Pilih 3 pembukaan yang paling sering Anda mainkan
- Atur bobot antara Content-Based dan Collaborative Filtering

### 2. Dapatkan Rekomendasi
- Klik tombol "Dapatkan Rekomendasi"
- Aplikasi akan memproses dan menampilkan hasil dalam 3 tab:
  - **Content-Based**: Berdasarkan kemiripan pembukaan
  - **Collaborative**: Berdasarkan pemain serupa
  - **Hybrid**: Kombinasi kedua metode

### 3. Analisis Detail
- Setiap rekomendasi menampilkan:
  - Nama pembukaan dan arketipenya
  - Langkah-langkah pembukaan
  - Skor rekomendasi dari masing-masing metode
  - Statistik tingkat kemenangan (jika tersedia)

## 🔧 Konfigurasi

### Parameter yang Dapat Disesuaikan

- **Alpha (α)**: Bobot untuk hybrid filtering
  - Nilai mendekati 1: Lebih mengutamakan Content-Based
  - Nilai mendekati 0: Lebih mengutamakan Collaborative
  - Default: 0.7

- **Top N**: Jumlah rekomendasi yang ditampilkan
  - Default: 5 rekomendasi per metode

## 🧠 Metodologi Machine Learning

### Content-Based Filtering
- Menganalisis kemiripan langkah pembukaan
- Menggunakan cosine similarity untuk menghitung kemiripan
- Memberikan skor berdasarkan kesamaan pola langkah

### Collaborative Filtering
- Menggunakan Neural Network dengan embedding
- Mencari pemain dengan rating serupa
- Memprediksi preferensi berdasarkan pola pemain lain
- Menyesuaikan kompleksitas pembukaan dengan rating

### Hybrid Filtering
- Menggabungkan kedua metode dengan pembobotan
- Normalisasi skor untuk keseimbangan
- Mengoptimalkan hasil berdasarkan parameter alpha

## 📈 Fitur Analisis

- **Profil Pembukaan Favorit**: Statistik pembukaan yang Anda pilih
- **Distribusi Rating**: Posisi rating Anda dalam dataset
- **Pembukaan Populer**: Daftar pembukaan yang paling sering dimainkan
- **Tingkat Kemenangan**: Statistik hasil untuk setiap pembukaan

## 🐛 Troubleshooting

### Masalah Umum

1. **Error "Model tidak dapat dimuat"**
   - Pastikan folder `models/` dan semua file model tersedia
   - Jalankan notebook pelatihan untuk menghasilkan model

2. **Dataset tidak ditemukan**
   - Pastikan file `games.csv` ada di direktori utama
   - Periksa format dan struktur data dalam CSV

3. **Memory Error**
   - Tutup aplikasi lain yang menggunakan memori besar
   - Gunakan sistem dengan minimum 4GB RAM

4. **Port sudah digunakan**
   ```bash
   streamlit run app.py --server.port 8502
   ```
