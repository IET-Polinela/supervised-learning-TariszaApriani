import pandas as pd

# Unggah file terlebih dahulu di Google Colab, lalu gunakan path yang sesuai
file_path = "train.csv"  # Sesuaikan dengan lokasi file

# Membaca dataset
df = pd.read_csv(file_path)

# Memilih hanya kolom numerik untuk analisis statistik
df_numeric = df.select_dtypes(include=['number'])

# Menghitung statistik deskriptif lengkap
stats_summary = df_numeric.describe().T

# Menambahkan median (Q2)
stats_summary['median'] = df_numeric.median()

# Menambahkan jumlah data yang tersedia (non-null count)
stats_summary['count'] = df_numeric.count()

# Menampilkan hasil
print(stats_summary[['count', 'mean', 'median', 'std', 'min', '25%', '50%', '75%', 'max']])

