import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Gunakan dataset tanpa outlier
df_scaled = df_no_outliers.copy()

# Pilih fitur numerik
numerical_features = df_scaled.select_dtypes(include=[np.number]).columns.tolist()

# Scaling dengan StandardScaler
scaler_standard = StandardScaler()
df_standard_scaled = pd.DataFrame(scaler_standard.fit_transform(df_scaled[numerical_features]), columns=numerical_features)

# Scaling dengan MinMaxScaler
scaler_minmax = MinMaxScaler()
df_minmax_scaled = pd.DataFrame(scaler_minmax.fit_transform(df_scaled[numerical_features]), columns=numerical_features)

# Visualisasi histogram sebelum dan sesudah scaling
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))
fig.suptitle("Distribusi Data Sebelum dan Sesudah Scaling", fontsize=16)

features_to_plot = numerical_features[:9]  # Pilih beberapa fitur untuk ditampilkan
for i, feature in enumerate(features_to_plot):
    row, col = i // 3, i % 3
    sns.histplot(df_scaled[feature], ax=axes[row, col], kde=True, color="blue", label="Original", alpha=0.5)
    sns.histplot(df_standard_scaled[feature], ax=axes[row, col], kde=True, color="red", label="StandardScaler", alpha=0.5)
    sns.histplot(df_minmax_scaled[feature], ax=axes[row, col], kde=True, color="green", label="MinMaxScaler", alpha=0.5)
    axes[row, col].set_title(f"Distribusi: {feature}")
    axes[row, col].legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])

# Simpan visualisasi
plt.savefig("feature_scaling.png", dpi=300)

plt.show()
