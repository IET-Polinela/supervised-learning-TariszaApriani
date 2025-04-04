import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Ambil hanya fitur numerik dan drop baris dengan nilai NaN
df_numerik = df.select_dtypes(include=[np.number]).dropna()

# Dataset A: dengan outlier
X_a = df_numerik.drop(columns=["SalePrice"])
y_a = df_numerik["SalePrice"]
X_a_train, X_a_test, y_a_train, y_a_test = train_test_split(X_a, y_a, test_size=0.2, random_state=42)

# Dataset B: tanpa outlier dan sudah distandarisasi
# Hilangkan outlier berdasarkan metode IQR
def remove_outliers(data, feature):
    Q1 = data[feature].quantile(0.25)
    Q3 = data[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return data[(data[feature] >= lower) & (data[feature] <= upper)]

df_b = df_numerik.copy()
for feature in df_b.columns:
    df_b = remove_outliers(df_b, feature)

X_b = df_b.drop(columns=["SalePrice"])
y_b = df_b["SalePrice"]

# Scaling
scaler = StandardScaler()
X_b_scaled = scaler.fit_transform(X_b)

# Train-test split
X_b_train, X_b_test, y_b_train, y_b_test = train_test_split(X_b_scaled, y_b, test_size=0.2, random_state=42)

# Linear Regression model
model_a = LinearRegression().fit(X_a_train, y_a_train)
model_b = LinearRegression().fit(X_b_train, y_b_train)

# Predictions
y_a_pred = model_a.predict(X_a_test)
y_b_pred = model_b.predict(X_b_test)

# Evaluation metrics
mse_a = mean_squared_error(y_a_test, y_a_pred)
r2_a = r2_score(y_a_test, y_a_pred)
mse_b = mean_squared_error(y_b_test, y_b_pred)
r2_b = r2_score(y_b_test, y_b_pred)

# Visualisasi
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle("Linear Regression Evaluation", fontsize=16)

# Scatter plot: Predicted vs Actual
axes[0, 0].scatter(y_a_test, y_a_pred, alpha=0.5)
axes[0, 0].set_title("A: Predicted vs Actual")
axes[0, 0].set_xlabel("Actual")
axes[0, 0].set_ylabel("Predicted")

axes[0, 1].scatter(y_b_test, y_b_pred, alpha=0.5)
axes[0, 1].set_title("B: Predicted vs Actual")
axes[0, 1].set_xlabel("Actual")
axes[0, 1].set_ylabel("Predicted")

# Residual plot
axes[1, 0].scatter(y_a_pred, y_a_test - y_a_pred, alpha=0.5)
axes[1, 0].axhline(0, color='red', linestyle='--')
axes[1, 0].set_title("A: Residual Plot")
axes[1, 0].set_xlabel("Predicted")
axes[1, 0].set_ylabel("Residual")

axes[1, 1].scatter(y_b_pred, y_b_test - y_b_pred, alpha=0.5)
axes[1, 1].axhline(0, color='red', linestyle='--')
axes[1, 1].set_title("B: Residual Plot")
axes[1, 1].set_xlabel("Predicted")
axes[1, 1].set_ylabel("Residual")

# Residual distribution
sns.histplot(y_a_test - y_a_pred, kde=True, ax=axes[2, 0])
axes[2, 0].set_title("A: Residual Distribution")

sns.histplot(y_b_test - y_b_pred, kde=True, ax=axes[2, 1])
axes[2, 1].set_title("B: Residual Distribution")

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Simpan visualisasi
plt.savefig("linear_regression.png", dpi=300)

plt.show()

(mse_a, r2_a, mse_b, r2_b)
