import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("train.csv")  # Ganti dengan nama file dataset Anda

# 1. Encoding fitur non-numerik
label_encoders = {}  # Dictionary untuk menyimpan encoder setiap kolom

for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))  # Ubah jadi numerik
    label_encoders[col] = le  # Simpan encoder untuk inverse transform nanti jika perlu

# 2. Memisahkan fitur independent (X) dan target (Y)
X = df.drop(columns=['SalePrice'])  # Ganti 'SalePrice' dengan nama target pada dataset Anda
Y = df['SalePrice']

# 3. Membagi dataset menjadi Training (80%) dan Testing (20%)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Cek hasil
print("Shape X_train:", X_train.shape)
print("Shape X_test:", X_test.shape)
print("Shape Y_train:", Y_train.shape)
print("Shape Y_test:", Y_test.shape)
