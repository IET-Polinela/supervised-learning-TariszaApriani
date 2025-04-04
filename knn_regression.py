from sklearn.neighbors import KNeighborsRegressor

# Fungsi evaluasi model KNN
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

# Visualisasi hasil prediksi KNN
plt.figure(figsize=(18, 5))
for i, k in enumerate([3, 5, 7]):
    plt.subplot(1, 3, i + 1)
    sns.scatterplot(x=results_knn[k]['y_true'], y=results_knn[k]['y_pred'])
    plt.title(f"KNN Regression (k={k})")
    plt.xlabel("Actual SalePrice")
    plt.ylabel("Predicted SalePrice")
    plt.grid(True)
plt.tight_layout()

# Simpan visualisasi sebagai file gambar
plt.savefig("knn_regression.png")

plt.show()

# Cetak hasil evaluasi
for k in [3, 5, 7]:
    print(f"K = {k}: MSE = {results_knn[k]['mse']:.2f}, R2 = {results_knn[k]['r2']:.4f}")
