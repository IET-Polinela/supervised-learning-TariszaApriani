import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
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

# Fungsi evaluasi Polynomial Regression
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

# Degree 2
mse2, r2_2, y_test2, y_pred2 = evaluate_polynomial_model(2)

# Degree 3
mse3, r2_3, y_test3, y_pred3 = evaluate_polynomial_model(3)

# Visualisasi
plt.figure(figsize=(16, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(x=y_test2, y=y_pred2)
plt.title("Polynomial Degree 2: Predicted vs Actual")
plt.xlabel("Actual SalePrice")
plt.ylabel("Predicted SalePrice")

plt.subplot(1, 2, 2)
sns.scatterplot(x=y_test3, y=y_pred3)
plt.title("Polynomial Degree 3: Predicted vs Actual")
plt.xlabel("Actual SalePrice")
plt.ylabel("Predicted SalePrice")

plt.tight_layout()

# Simpan gambar
plt.savefig("polynomial_regression.png")

# Tampilkan plot
plt.show()

(mse2, r2_2, mse3, r2_3)
