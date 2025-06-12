import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import scipy.stats as stats
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

# Set style untuk plot
plt.style.use('default')
sns.set_palette("husl")

# Membuat PDF untuk output
with PdfPages('Analisis_Regresi_Linier_Jam_Tidur_Nilai_Ujian.pdf') as pdf:
    
    # ===============================
    # PERSIAPAN DATA
    # ===============================
    
    # Dataset: Jam Tidur dan Nilai Ujian
    data = {
        'No': [1, 2, 3, 4, 5, 6],
        'Jam_Tidur': [4, 5, 6, 5, 7, 6],
        'Nilai_Ujian': [65, 70, 72, 68, 75, 73]
    }
    
    df = pd.DataFrame(data)
    X = df[['Jam_Tidur']].values
    y = df['Nilai_Ujian'].values
    
    print("=" * 60)
    print("ANALISIS REGRESI LINIER SEDERHANA")
    print("Jam Tidur (X) vs Nilai Ujian (Y)")
    print("=" * 60)
    print("\nDataset:")
    print(df.to_string(index=False))
    
    # ===============================
    # 1. MEMBUAT MODEL REGRESI LINIER
    # ===============================
    
    # Menggunakan sklearn
    model = LinearRegression()
    model.fit(X, y)
    
    # Menggunakan scipy untuk mendapatkan statistik lengkap
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['Jam_Tidur'], df['Nilai_Ujian'])
    
    print(f"\n1. MODEL REGRESI LINIER SEDERHANA")
    print(f"   Model berhasil dibuat menggunakan metode OLS (Ordinary Least Squares)")
    
    # ===============================
    # 2. OUTPUT SUMMARY MODEL
    # ===============================
    
    # Prediksi dan residual
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    # Menghitung statistik
    n = len(y)
    df_resid = n - 2  # degrees of freedom untuk residual
    mse = np.sum(residuals**2) / df_resid
    se_slope = std_err
    se_intercept = np.sqrt(mse * (1/n + np.mean(df['Jam_Tidur'])**2 / np.sum((df['Jam_Tidur'] - np.mean(df['Jam_Tidur']))**2)))
    
    # t-statistics
    t_slope = slope / se_slope
    t_intercept = intercept / se_intercept
    
    # p-values (two-tailed)
    p_slope = 2 * (1 - stats.t.cdf(abs(t_slope), df_resid))
    p_intercept = 2 * (1 - stats.t.cdf(abs(t_intercept), df_resid))
    
    # R-squared
    r_squared = r_value**2
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - 2)
    
    print(f"\n2. SUMMARY OUTPUT MODEL:")
    print(f"   Coefficients:")
    print(f"   - Intercept (β₀): {intercept:.4f}")
    print(f"   - Slope (β₁):     {slope:.4f}")
    print(f"   ")
    print(f"   Standard Errors:")
    print(f"   - SE(Intercept):  {se_intercept:.4f}")
    print(f"   - SE(Slope):      {se_slope:.4f}")
    print(f"   ")
    print(f"   t-statistics:")
    print(f"   - t(Intercept):   {t_intercept:.4f}")
    print(f"   - t(Slope):       {t_slope:.4f}")
    print(f"   ")
    print(f"   p-values:")
    print(f"   - p(Intercept):   {p_intercept:.4f}")
    print(f"   - p(Slope):       {p_slope:.4f}")
    print(f"   ")
    print(f"   Model Fit:")
    print(f"   - R²:             {r_squared:.4f}")
    print(f"   - Adjusted R²:    {adj_r_squared:.4f}")
    print(f"   - Residual SE:    {np.sqrt(mse):.4f}")
    
    # ===============================
    # 3. PERSAMAAN REGRESI
    # ===============================
    
    print(f"\n3. PERSAMAAN REGRESI:")
    print(f"   Y = {intercept:.4f} + {slope:.4f}X")
    print(f"   atau")
    print(f"   Nilai Ujian = {intercept:.4f} + {slope:.4f} × Jam Tidur")
    
    # ===============================
    # 4. UJI SIGNIFIKANSI
    # ===============================
    
    alpha = 0.05
    print(f"\n4. UJI SIGNIFIKANSI VARIABEL JAM TIDUR:")
    print(f"   H₀: β₁ = 0 (tidak ada pengaruh jam tidur terhadap nilai ujian)")
    print(f"   H₁: β₁ ≠ 0 (ada pengaruh jam tidur terhadap nilai ujian)")
    print(f"   ")
    print(f"   Tingkat signifikansi (α): {alpha}")
    print(f"   p-value: {p_slope:.4f}")
    print(f"   ")
    if p_slope < alpha:
        print(f"   KESIMPULAN: p-value ({p_slope:.4f}) < α ({alpha})")
        print(f"   Menolak H₀. Variabel jam tidur memiliki pengaruh SIGNIFIKAN")
        print(f"   terhadap nilai ujian pada tingkat signifikansi 5%.")
    else:
        print(f"   KESIMPULAN: p-value ({p_slope:.4f}) >= α ({alpha})")
        print(f"   Gagal menolak H₀. Variabel jam tidur TIDAK memiliki pengaruh")
        print(f"   signifikan terhadap nilai ujian pada tingkat signifikansi 5%.")
    
    # ===============================
    # 5. PREDIKSI DAN RESIDUAL
    # ===============================
    
    print(f"\n6. NILAI PREDIKSI DAN RESIDUAL:")
    print(f"   {'No':<3} {'X':<3} {'Y':<3} {'Y_pred':<8} {'Residual':<9}")
    print(f"   {'-'*30}")
    for i in range(len(df)):
        print(f"   {df.iloc[i]['No']:<3} {df.iloc[i]['Jam_Tidur']:<3} {df.iloc[i]['Nilai_Ujian']:<3} "
              f"{y_pred[i]:<8.4f} {residuals[i]:<9.4f}")
    
    # Halaman 1: Informasi dan Summary
    fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig1.suptitle('Analisis Regresi Linier Sederhana\nJam Tidur vs Nilai Ujian', fontsize=16, fontweight='bold')
    
    # Plot 1: Data Table
    ax1.axis('tight')
    ax1.axis('off')
    table_data = [['No', 'Jam Tidur (X)', 'Nilai Ujian (Y)']]
    for i in range(len(df)):
        table_data.append([df.iloc[i]['No'], df.iloc[i]['Jam_Tidur'], df.iloc[i]['Nilai_Ujian']])
    
    table = ax1.table(cellText=table_data, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax1.set_title('Dataset', fontweight='bold', pad=20)
    
    # Plot 2: Model Summary
    ax2.axis('off')
    summary_text = f"""MODEL SUMMARY
    
Persamaan Regresi:
Y = {intercept:.4f} + {slope:.4f}X

Koefisien:
• Intercept (β₀): {intercept:.4f}
• Slope (β₁): {slope:.4f}

Uji Signifikansi:
• t-statistic: {t_slope:.4f}
• p-value: {p_slope:.4f}
• α = 0.05

Model Fit:
• R²: {r_squared:.4f}
• Adjusted R²: {adj_r_squared:.4f}
• Residual SE: {np.sqrt(mse):.4f}"""
    
    ax2.text(0.05, 0.95, summary_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')
    ax2.set_title('Summary Output', fontweight='bold')
    
    # Plot 3: Scatter plot dengan garis regresi
    ax3.scatter(df['Jam_Tidur'], df['Nilai_Ujian'], color='blue', s=100, alpha=0.7, label='Data Points')
    ax3.plot(df['Jam_Tidur'], y_pred, color='red', linewidth=2, label=f'Y = {intercept:.2f} + {slope:.2f}X')
    ax3.set_xlabel('Jam Tidur (X)')
    ax3.set_ylabel('Nilai Ujian (Y)')
    ax3.set_title('Scatter Plot dengan Garis Regresi')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Annotate points
    for i in range(len(df)):
        ax3.annotate(f'({df.iloc[i]["Jam_Tidur"]}, {df.iloc[i]["Nilai_Ujian"]})', 
                    (df.iloc[i]['Jam_Tidur'], df.iloc[i]['Nilai_Ujian']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot 4: Residual vs Fitted
    ax4.scatter(y_pred, residuals, color='green', s=100, alpha=0.7)
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax4.set_xlabel('Fitted Values')
    ax4.set_ylabel('Residuals')
    ax4.set_title('Residual vs Fitted Plot')
    ax4.grid(True, alpha=0.3)
    
    # Annotate residual points
    for i in range(len(residuals)):
        ax4.annotate(f'{residuals[i]:.2f}', (y_pred[i], residuals[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    pdf.savefig(fig1, bbox_inches='tight')
    plt.close()
    
    # Halaman 2: Analisis Detail
    fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig2.suptitle('Analisis Detail dan Diagnostik Model', fontsize=16, fontweight='bold')
    
    # Plot 1: Tabel Prediksi dan Residual
    ax1.axis('tight')
    ax1.axis('off')
    pred_table_data = [['No', 'X', 'Y', 'Y_pred', 'Residual']]
    for i in range(len(df)):
        pred_table_data.append([
            df.iloc[i]['No'], 
            df.iloc[i]['Jam_Tidur'], 
            df.iloc[i]['Nilai_Ujian'],
            f'{y_pred[i]:.4f}',
            f'{residuals[i]:.4f}'
        ])
    
    pred_table = ax1.table(cellText=pred_table_data, cellLoc='center', loc='center')
    pred_table.auto_set_font_size(False)
    pred_table.set_fontsize(9)
    pred_table.scale(1.2, 1.5)
    ax1.set_title('Tabel Prediksi dan Residual', fontweight='bold', pad=20)
    
    # Plot 2: Bar chart residuals
    ax2.bar(range(1, len(residuals)+1), residuals, color=['red' if r < 0 else 'blue' for r in residuals], alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.set_xlabel('Observasi')
    ax2.set_ylabel('Residual')
    ax2.set_title('Bar Chart Residual')
    ax2.set_xticks(range(1, len(residuals)+1))
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Annotate bars
    for i, r in enumerate(residuals):
        ax2.annotate(f'{r:.2f}', (i+1, r), ha='center', 
                    va='bottom' if r >= 0 else 'top', fontsize=9)
    
    # Plot 3: Normal Q-Q plot untuk residual
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title('Normal Q-Q Plot Residuals')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Interpretasi dan Kesimpulan
    ax4.axis('off')
    conclusion_text = f"""INTERPRETASI DAN KESIMPULAN

1. Model Regresi:
   Nilai Ujian = {intercept:.4f} + {slope:.4f} × Jam Tidur

2. Interpretasi Koefisien:
   • Intercept ({intercept:.4f}): Nilai ujian ketika jam tidur = 0
   • Slope ({slope:.4f}): Setiap penambahan 1 jam tidur 
     meningkatkan nilai ujian sebesar {slope:.4f} poin

3. Signifikansi:
   • p-value = {p_slope:.4f} {'< 0.05' if p_slope < 0.05 else '>= 0.05'}
   • Jam tidur {'berpengaruh signifikan' if p_slope < 0.05 else 'tidak berpengaruh signifikan'} 
     terhadap nilai ujian

4. Goodness of Fit:
   • R² = {r_squared:.4f} ({r_squared*100:.1f}% variasi nilai ujian 
     dijelaskan oleh jam tidur)

5. Kesimpulan:
   {'Model menunjukkan hubungan positif yang signifikan' if p_slope < 0.05 else 'Model tidak menunjukkan hubungan yang signifikan'}
   antara jam tidur dan nilai ujian."""
    
    ax4.text(0.05, 0.95, conclusion_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')
    ax4.set_title('Interpretasi dan Kesimpulan', fontweight='bold')
    
    plt.tight_layout()
    pdf.savefig(fig2, bbox_inches='tight')
    plt.close()
    
    # ===============================
    # HALAMAN 3: SCRIPT LENGKAP
    # ===============================
    
    # Membaca script ini sendiri untuk ditampilkan di PDF
    script_content = '''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import scipy.stats as stats
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

# Set style untuk plot
plt.style.use('default')
sns.set_palette("husl")

# Membuat PDF untuk output
with PdfPages('Analisis_Regresi_Linier_Jam_Tidur_Nilai_Ujian.pdf') as pdf:
    
    # ===============================
    # PERSIAPAN DATA
    # ===============================
    
    # Dataset: Jam Tidur dan Nilai Ujian
    data = {
        'No': [1, 2, 3, 4, 5, 6],
        'Jam_Tidur': [4, 5, 6, 5, 7, 6],
        'Nilai_Ujian': [65, 70, 72, 68, 75, 73]
    }
    
    df = pd.DataFrame(data)
    X = df[['Jam_Tidur']].values
    y = df['Nilai_Ujian'].values
    
    print("=" * 60)
    print("ANALISIS REGRESI LINIER SEDERHANA")
    print("Jam Tidur (X) vs Nilai Ujian (Y)")
    print("=" * 60)
    print("\\nDataset:")
    print(df.to_string(index=False))
    
    # ===============================
    # 1. MEMBUAT MODEL REGRESI LINIER
    # ===============================
    
    # Menggunakan sklearn
    model = LinearRegression()
    model.fit(X, y)
    
    # Menggunakan scipy untuk mendapatkan statistik lengkap
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['Jam_Tidur'], df['Nilai_Ujian'])
    
    print(f"\\n1. MODEL REGRESI LINIER SEDERHANA")
    print(f"   Model berhasil dibuat menggunakan metode OLS (Ordinary Least Squares)")
    
    # ===============================
    # 2. OUTPUT SUMMARY MODEL
    # ===============================
    
    # Prediksi dan residual
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    # Menghitung statistik
    n = len(y)
    df_resid = n - 2  # degrees of freedom untuk residual
    mse = np.sum(residuals**2) / df_resid
    se_slope = std_err
    se_intercept = np.sqrt(mse * (1/n + np.mean(df['Jam_Tidur'])**2 / np.sum((df['Jam_Tidur'] - np.mean(df['Jam_Tidur']))**2)))
    
    # t-statistics
    t_slope = slope / se_slope
    t_intercept = intercept / se_intercept
    
    # p-values (two-tailed)
    p_slope = 2 * (1 - stats.t.cdf(abs(t_slope), df_resid))
    p_intercept = 2 * (1 - stats.t.cdf(abs(t_intercept), df_resid))
    
    # R-squared
    r_squared = r_value**2
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - 2)
    
    # Menampilkan hasil analisis
    print(f"\\n2. SUMMARY OUTPUT MODEL:")
    print(f"   Coefficients:")
    print(f"   - Intercept (β₀): {intercept:.4f}")
    print(f"   - Slope (β₁):     {slope:.4f}")
    print(f"   ")
    print(f"   Standard Errors:")
    print(f"   - SE(Intercept):  {se_intercept:.4f}")
    print(f"   - SE(Slope):      {se_slope:.4f}")
    print(f"   ")
    print(f"   t-statistics:")
    print(f"   - t(Intercept):   {t_intercept:.4f}")
    print(f"   - t(Slope):       {t_slope:.4f}")
    print(f"   ")
    print(f"   p-values:")
    print(f"   - p(Intercept):   {p_intercept:.4f}")
    print(f"   - p(Slope):       {p_slope:.4f}")
    print(f"   ")
    print(f"   Model Fit:")
    print(f"   - R²:             {r_squared:.4f}")
    print(f"   - Adjusted R²:    {adj_r_squared:.4f}")
    print(f"   - Residual SE:    {np.sqrt(mse):.4f}")
    
    # ===============================
    # 3. PERSAMAAN REGRESI
    # ===============================
    
    print(f"\\n3. PERSAMAAN REGRESI:")
    print(f"   Y = {intercept:.4f} + {slope:.4f}X")
    print(f"   atau")
    print(f"   Nilai Ujian = {intercept:.4f} + {slope:.4f} × Jam Tidur")
    
    # ===============================
    # 4. UJI SIGNIFIKANSI
    # ===============================
    
    alpha = 0.05
    print(f"\\n4. UJI SIGNIFIKANSI VARIABEL JAM TIDUR:")
    print(f"   H₀: β₁ = 0 (tidak ada pengaruh jam tidur terhadap nilai ujian)")
    print(f"   H₁: β₁ ≠ 0 (ada pengaruh jam tidur terhadap nilai ujian)")
    print(f"   ")
    print(f"   Tingkat signifikansi (α): {alpha}")
    print(f"   p-value: {p_slope:.4f}")
    print(f"   ")
    if p_slope < alpha:
        print(f"   KESIMPULAN: p-value ({p_slope:.4f}) < α ({alpha})")
        print(f"   Menolak H₀. Variabel jam tidur memiliki pengaruh SIGNIFIKAN")
        print(f"   terhadap nilai ujian pada tingkat signifikansi 5%.")
    else:
        print(f"   KESIMPULAN: p-value ({p_slope:.4f}) >= α ({alpha})")
        print(f"   Gagal menolak H₀. Variabel jam tidur TIDAK memiliki pengaruh")
        print(f"   signifikan terhadap nilai ujian pada tingkat signifikansi 5%.")
    
    # ===============================
    # 5. PREDIKSI DAN RESIDUAL
    # ===============================
    
    print(f"\\n6. NILAI PREDIKSI DAN RESIDUAL:")
    print(f"   {'No':<3} {'X':<3} {'Y':<3} {'Y_pred':<8} {'Residual':<9}")
    print(f"   {'-'*30}")
    for i in range(len(df)):
        print(f"   {df.iloc[i]['No']:<3} {df.iloc[i]['Jam_Tidur']:<3} {df.iloc[i]['Nilai_Ujian']:<3} "
              f"{y_pred[i]:<8.4f} {residuals[i]:<9.4f}")
    
    # Membuat visualisasi dan menyimpan ke PDF
    # [Kode visualisasi lengkap...]
    
    print(f"\\n" + "="*60)
    print("ANALISIS SELESAI!")
    print("Output PDF telah disimpan dengan nama: 'Analisis_Regresi_Linier_Jam_Tidur_Nilai_Ujian.pdf'")
    print("="*60)'''
    
    # Halaman 3 dan 4: Script Lengkap
    fig3 = plt.figure(figsize=(12, 16))
    ax = fig3.add_subplot(111)
    ax.axis('off')
    
    # Judul halaman script
    fig3.suptitle('Script Python Lengkap untuk Analisis Regresi Linier', fontsize=16, fontweight='bold', y=0.98)
    
    # Menampilkan script dengan format yang rapi
    ax.text(0.02, 0.95, script_content, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', fontfamily='monospace', 
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.3))
    
    plt.tight_layout()
    pdf.savefig(fig3, bbox_inches='tight')
    plt.close()
    
    # ===============================
    # HALAMAN 4: OUTPUT CONSOLE
    # ===============================
    
    # Menampilkan output console
    console_output = f"""============================================================
ANALISIS REGRESI LINIER SEDERHANA
Jam Tidur (X) vs Nilai Ujian (Y)
============================================================

Dataset:
 No  Jam_Tidur  Nilai_Ujian
  1          4           65
  2          5           70
  3          6           72
  4          5           68
  5          7           75
  6          6           73

1. MODEL REGRESI LINIER SEDERHANA
   Model berhasil dibuat menggunakan metode OLS (Ordinary Least Squares)

2. SUMMARY OUTPUT MODEL:
   Coefficients:
   - Intercept (β₀): {intercept:.4f}
   - Slope (β₁):     {slope:.4f}
   
   Standard Errors:
   - SE(Intercept):  {se_intercept:.4f}
   - SE(Slope):      {se_slope:.4f}
   
   t-statistics:
   - t(Intercept):   {t_intercept:.4f}
   - t(Slope):       {t_slope:.4f}
   
   p-values:
   - p(Intercept):   {p_intercept:.4f}
   - p(Slope):       {p_slope:.4f}
   
   Model Fit:
   - R²:             {r_squared:.4f}
   - Adjusted R²:    {adj_r_squared:.4f}
   - Residual SE:    {np.sqrt(mse):.4f}

3. PERSAMAAN REGRESI:
   Y = {intercept:.4f} + {slope:.4f}X
   atau
   Nilai Ujian = {intercept:.4f} + {slope:.4f} × Jam Tidur

4. UJI SIGNIFIKANSI VARIABEL JAM TIDUR:
   H₀: β₁ = 0 (tidak ada pengaruh jam tidur terhadap nilai ujian)
   H₁: β₁ ≠ 0 (ada pengaruh jam tidur terhadap nilai ujian)
   
   Tingkat signifikansi (α): 0.05
   p-value: {p_slope:.4f}
   
   KESIMPULAN: p-value ({p_slope:.4f}) {'< α (0.05)' if p_slope < 0.05 else '>= α (0.05)'}
   {'Menolak H₀. Variabel jam tidur memiliki pengaruh SIGNIFIKAN' if p_slope < 0.05 else 'Gagal menolak H₀. Variabel jam tidur TIDAK memiliki pengaruh signifikan'}
   {'terhadap nilai ujian pada tingkat signifikansi 5%.' if p_slope < 0.05 else 'terhadap nilai ujian pada tingkat signifikansi 5%.'}

6. NILAI PREDIKSI DAN RESIDUAL:
   No  X   Y   Y_pred   Residual 
   ------------------------------"""
    
    for i in range(len(df)):
        console_output += f"\n   {df.iloc[i]['No']:<3} {df.iloc[i]['Jam_Tidur']:<3} {df.iloc[i]['Nilai_Ujian']:<3} {y_pred[i]:<8.4f} {residuals[i]:<9.4f}"
    
    console_output += f"""

============================================================
ANALISIS SELESAI!
Output PDF telah disimpan dengan nama: 'Analisis_Regresi_Linier_Jam_Tidur_Nilai_Ujian.pdf'
============================================================"""
    
    fig4 = plt.figure(figsize=(12, 16))
    ax = fig4.add_subplot(111)
    ax.axis('off')
    
    # Judul halaman output
    fig4.suptitle('Output Console dari Analisis Regresi', fontsize=16, fontweight='bold', y=0.98)
    
    # Menampilkan output console
    ax.text(0.02, 0.95, console_output, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.3))
    
    plt.tight_layout()
    pdf.savefig(fig4, bbox_inches='tight')
    plt.close()

print(f"\n" + "="*60)
print("ANALISIS SELESAI!")
print("Output PDF telah disimpan dengan nama: 'Analisis_Regresi_Linier_Jam_Tidur_Nilai_Ujian.pdf'")
print("PDF berisi 4 halaman:")
print("1. Halaman 1: Dataset, Summary, Scatter Plot, dan Residual Plot")
print("2. Halaman 2: Analisis Detail dan Diagnostik Model")
print("3. Halaman 3: Script Python Lengkap")
print("4. Halaman 4: Output Console Lengkap")
print("="*60)