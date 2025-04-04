import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = "train.csv"  # Sesuaikan path jika berbeda
df = pd.read_csv(file_path)

# Menampilkan informasi awal dataset
print(df.info())
print(df.describe())

# Identifikasi outlier menggunakan metode IQR
def detect_outliers_iqr(data, feature):
    Q1 = data[feature].quantile(0.25)
    Q3 = data[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[feature] < lower_bound) | (data[feature] > upper_bound)]

# Identifikasi outlier untuk semua fitur numerik
numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
outliers = {feature: detect_outliers_iqr(df, feature) for feature in numerical_features}

# Menghapus outlier dari dataset
def remove_outliers(data, feature):
    Q1 = data[feature].quantile(0.25)
    Q3 = data[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[feature] >= lower_bound) & (data[feature] <= upper_bound)]

df_no_outliers = df.copy()
for feature in numerical_features:
    df_no_outliers = remove_outliers(df_no_outliers, feature)

# Visualisasi Boxplot Sebelum Penghapusan Outlier
plt.figure(figsize=(20, 60))
for i, col in enumerate(numerical_features, 1):
    plt.subplot(len(numerical_features)//4 + 1, 4, i)
    sns.boxplot(x=df[col])
    plt.title(f"Before - {col}")
plt.tight_layout()
plt.savefig("outlier_before.png", dpi=300)  # Simpan gambar sebelum outlier dihapus
plt.show()

# Visualisasi Boxplot Sesudah Penghapusan Outlier
plt.figure(figsize=(20, 60))
for i, col in enumerate(numerical_features, 1):
    plt.subplot(len(numerical_features)//4 + 1, 4, i)
    sns.boxplot(x=df_no_outliers[col])
    plt.title(f"After - {col}")
plt.tight_layout()
plt.savefig("outlier_after.png", dpi=300)  # Simpan gambar sesudah outlier dihapus
plt.show()

# Output jumlah data sebelum dan sesudah penghapusan outlier
print("Sebelum penghapusan outlier:", df.shape)
print("Sesudah penghapusan outlier:", df_no_outliers.shape)