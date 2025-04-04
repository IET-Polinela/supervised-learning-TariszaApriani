import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("train.csv")

# Hapus outlier dari semua fitur numerik
def remove_outliers(data, feature):
    Q1 = data[feature].quantile(0.25)
    Q3 = data[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[feature] >= lower_bound) & (data[feature] <= upper_bound)]

numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
numerical_features.remove("SalePrice")
df_no_outliers = df.copy()
for feature in numerical_features:
    df_no_outliers = remove_outliers(df_no_outliers, feature)

X = df_no_outliers[numerical_features]
y = df_no_outliers['SalePrice']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Evaluasi Polynomial Regression
def evaluate_polynomial_model(degree):
    poly = PolynomialFeatures(degree=degree)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_poly_train, y_train)
    y_pred = model.predict(X_poly_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mse, r2, y_test, y_pred

# Degree 2 & 3
mse2, r2_2, y_test2, y_pred2 = evaluate_polynomial_model(2)
mse3, r2_3, y_test3, y_pred3 = evaluate_polynomial_model(3)

# Evaluasi Linear Regression
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
lr_pred = model_lr.predict(X_test)

mse_lr = mean_squared_error(y_test, lr_pred)
r2_lr = r2_score(y_test, lr_pred)

print(f"Linear Regression - MSE: {mse_lr:.2f}, R2: {r2_lr:.4f}")

# Evaluasi KNN Regression
def evaluate_knn_model(k):
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mse, r2, y_test, y_pred

# Uji dengan K = 3, 5, 7
results_knn = {}
for k in [3, 5, 7]:
    mse, r2, y_true, y_pred = evaluate_knn_model(k)
    results_knn[k] = {'mse': mse, 'r2': r2, 'y_true': y_true, 'y_pred': y_pred}

# Perbandingan Model
comparison_results = {
    "Linear Regression": {"MSE": mse_lr, "R2": r2_lr},
    "Polynomial Degree 2": {"MSE": mse2, "R2": r2_2},
    "Polynomial Degree 3": {"MSE": mse3, "R2": r2_3},
    "KNN (k=3)": {"MSE": results_knn[3]['mse'], "R2": results_knn[3]['r2']},
    "KNN (k=5)": {"MSE": results_knn[5]['mse'], "R2": results_knn[5]['r2']},
    "KNN (k=7)": {"MSE": results_knn[7]['mse'], "R2": results_knn[7]['r2']}
}

# Tampilkan sebagai tabel
comparison_df = pd.DataFrame(comparison_results).T
print("Perbandingan Model:")
print(comparison_df)

# Visualisasi hasil prediksi semua model
plt.figure(figsize=(18, 12))

# Linear Regression
plt.subplot(2, 3, 1)
sns.scatterplot(x=y_test, y=lr_pred)
plt.title("Linear Regression")
plt.xlabel("Actual")
plt.ylabel("Predicted")

# Polynomial Degree 2
plt.subplot(2, 3, 2)
sns.scatterplot(x=y_test2, y=y_pred2)
plt.title("Polynomial Degree 2")
plt.xlabel("Actual")
plt.ylabel("Predicted")

# Polynomial Degree 3
plt.subplot(2, 3, 3)
sns.scatterplot(x=y_test3, y=y_pred3)
plt.title("Polynomial Degree 3")
plt.xlabel("Actual")
plt.ylabel("Predicted")

# KNN K=3
plt.subplot(2, 3, 4)
sns.scatterplot(x=results_knn[3]['y_true'], y=results_knn[3]['y_pred'])
plt.title("KNN (k=3)")
plt.xlabel("Actual")
plt.ylabel("Predicted")

# KNN K=5
plt.subplot(2, 3, 5)
sns.scatterplot(x=results_knn[5]['y_true'], y=results_knn[5]['y_pred'])
plt.title("KNN (k=5)")
plt.xlabel("Actual")
plt.ylabel("Predicted")

# KNN K=7
plt.subplot(2, 3, 6)
sns.scatterplot(x=results_knn[7]['y_true'], y=results_knn[7]['y_pred'])
plt.title("KNN (k=7)")
plt.xlabel("Actual")
plt.ylabel("Predicted")

plt.tight_layout()

# Simpan visualisasi
plt.savefig("model_comparison.png")

plt.show()
