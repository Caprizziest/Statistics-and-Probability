import numpy as np
from scipy import stats

# Fungsi untuk uji hipotesis regresi linier sederhana
def uji_regresi(X, Y, judul):
    n = len(X)
    
    # 1. Hitung rata-rata X dan Y
    X_bar = np.mean(X)
    Y_bar = np.mean(Y)

    # 2. Hitung koefisien regresi beta1 dan beta0
    numerator = sum((X - X_bar) * (Y - Y_bar))
    denominator = sum((X - X_bar)**2)
    beta1_hat = numerator / denominator
    beta0_hat = Y_bar - beta1_hat * X_bar

    # 3. Model regresi: prediksi Y
    Y_pred = beta0_hat + beta1_hat * X

    # 4. Hitung residual dan varians galat
    e = Y - Y_pred
    sigma2_hat = sum(e**2) / (n - 2)

    # 5. Standard Error beta1
    SE_beta1 = np.sqrt(sigma2_hat / denominator)

    # 6. Statistik uji t
    t_stat = beta1_hat / SE_beta1

    # 7. P-value (dua arah)
    df = n - 2
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=df))

    # 8. Output hasil
    print(f"\n{'='*50}")
    print(f"{judul}")
    print(f"{'='*50}")
    print("Rata-rata X:", X_bar)
    print("Rata-rata Y:", Y_bar)
    print("Koefisien β1:", beta1_hat)
    print("Intercept β0:", beta0_hat)
    print("Model Regresi: Y = {:.2f} + {:.2f} * X".format(beta0_hat, beta1_hat))
    print("Prediksi Y:", Y_pred)
    print("Residual ei:", e)
    print("Varians Galat σ²:", sigma2_hat)
    print("Standard Error β1:", SE_beta1)
    print("Statistik t:", t_stat)
    print("P-Value:", p_value)
    
    alpha = 0.05
    if p_value < alpha:
        print("Kesimpulan: Ada pengaruh signifikan dari X terhadap Y.")
    else:
        print("Kesimpulan: Tidak ada pengaruh signifikan dari X terhadap Y.")
    print(f"{'='*50}")

# Data Soal 1
X1 = np.array([4, 5, 6, 5, 7, 6])  # Jam Tidur
Y1 = np.array([65, 70, 72, 68, 75, 73])  # Nilai Ujian

# Data Soal 2
X2 = np.array([1, 3, 2, 4, 5])  # Jam Belajar
Y2 = np.array([55, 60, 58, 67, 73])  # Nilai Matematika

# Jalankan analisis
uji_regresi(X1, Y1, "Soal 1: Pengaruh Jam Tidur terhadap Nilai Ujian")
uji_regresi(X2, Y2, "Soal 2: Pengaruh Jam Belajar terhadap Nilai Matematika")