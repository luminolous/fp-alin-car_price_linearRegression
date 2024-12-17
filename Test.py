import numpy as np
from scipy.optimize import linprog
import pandas as pd

# Data mobil dari hasil regresi
car_data = pd.DataFrame({
    'Car Model': ['Model_A', 'Model_B', 'Model_C'],
    'Predicted Price': [25000, 30000, 20000],  # Harga prediksi
    'Curbweight': [1.5, 1.8, 1.3],  # Berat mobil dalam ton
    'Highway MPG': [15, 12, 18]  # Efisiensi bahan bakar dalam km/l
})

# Mengambil data dari DataFrame
car_models = car_data['Car Model'].tolist()
predicted_prices = car_data['Predicted Price'].tolist()
weights = car_data['Curbweight'].tolist()
fuel_efficiencies = car_data['Highway MPG'].tolist()

# Batasan (disesuaikan lebih lanjut)
max_weight_capacity = 20  # Kapasitas berat maksimum (ton)
min_fuel_efficiency = 10  # Efisiensi bahan bakar minimum (km/l)
max_production_capacity = 100  # Kapasitas produksi maksimum

# Fungsi tujuan: Maksimalkan keuntungan
objective = -np.array(predicted_prices)  # Negatif untuk maksimasi

# Matriks batasan dalam bentuk Ax <= b
A = [
    weights,  # Total berat <= kapasitas berat maksimum
    [-eff for eff in fuel_efficiencies],  # Efisiensi bahan bakar rata-rata >= minimum
    [1, 1, 1],  # Total produksi <= kapasitas produksi maksimum
]
b = [
    max_weight_capacity,  # Kapasitas berat maksimum
    -min_fuel_efficiency * max_production_capacity,  # Efisiensi bahan bakar minimum
    max_production_capacity,  # Kapasitas produksi maksimum
]

# Batasan untuk setiap variabel keputusan (produksi tidak boleh negatif)
x_bounds = [(0, None) for _ in car_models]

# Menjalankan optimasi Linear Programming
result = linprog(c=objective, A_ub=A, b_ub=b, bounds=x_bounds, method='highs')

# Menampilkan hasil
if result.success:
    print("Rencana produksi optimal:")
    for i, model in enumerate(car_models):
        print(f"{model}: {result.x[i]:.2f} unit")
    print(f"Keuntungan Maksimum: ${-result.fun:.2f}")
else:
    print("Optimasi gagal:", result.message)
