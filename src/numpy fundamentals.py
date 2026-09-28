import numpy as np


print("=" * 60)
print("PYTHON FOR ML - NUMPY FUNDAMENTALS")
print("=" * 60)


# ============================================================
# 1. CREATE 1D, 2D AND 3D ARRAYS
# ============================================================

array_1d = np.array([1, 2, 3, 4, 5])

array_2d = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

array_3d = np.array([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
])

print("\n1D ARRAY:")
print(array_1d)
print("Shape:", array_1d.shape)

print("\n2D ARRAY:")
print(array_2d)
print("Shape:", array_2d.shape)

print("\n3D ARRAY:")
print(array_3d)
print("Shape:", array_3d.shape)


# ============================================================
# 2. BROADCASTING
# ============================================================

matrix = np.array([
    [10, 20, 30],
    [40, 50, 60]
])

addition_vector = np.array([1, 2, 3])

broadcast_result = matrix + addition_vector

print("\nBROADCASTING:")
print("Original matrix:")
print(matrix)

print("Vector:")
print(addition_vector)

print("After broadcasting:")
print(broadcast_result)


# ============================================================
# 3. VECTORIZED OPERATIONS
# ============================================================

numbers = np.array([1, 2, 3, 4, 5])

squared = numbers ** 2
multiplied = numbers * 10
normalized = numbers / np.max(numbers)

print("\nVECTORIZED OPERATIONS:")
print("Original:", numbers)
print("Squared:", squared)
print("Multiplied by 10:", multiplied)
print("Normalized:", normalized)


# ============================================================
# 4. MATRIX MULTIPLICATION
# ============================================================

matrix_a = np.array([
    [1, 2],
    [3, 4]
])

matrix_b = np.array([
    [5, 6],
    [7, 8]
])

matrix_product = matrix_a @ matrix_b

print("\nMATRIX MULTIPLICATION:")
print("Matrix A:")
print(matrix_a)

print("Matrix B:")
print(matrix_b)

print("A @ B:")
print(matrix_product)


# ============================================================
# 5. LOAD REAL IRIS CSV DATASET
# ============================================================

# The CSV should be located at:
# AIML-INTERNSHIP/data/iris.csv

data_path = "data/iris.csv"

# Iris CSV contains:
# sepal_length, sepal_width, petal_length, petal_width, species
#
# We only need the four numerical columns for NumPy statistics.

data = np.loadtxt(
    data_path,
    delimiter=",",
    skiprows=1,
    usecols=(0, 1, 2, 3)
)

print("\nIRIS DATASET:")
print("Dataset shape:", data.shape)
print("First 5 rows:")
print(data[:5])


# ============================================================
# 6. MEAN
# ============================================================

mean_values = np.mean(data, axis=0)

print("\nMEAN:")
print(mean_values)


# ============================================================
# 7. STANDARD DEVIATION
# ============================================================

std_values = np.std(data, axis=0)

print("\nSTANDARD DEVIATION:")
print(std_values)


# ============================================================
# 8. CORRELATION MATRIX
# ============================================================

correlation_matrix = np.corrcoef(data, rowvar=False)

print("\nCORRELATION MATRIX:")
print(correlation_matrix)


# ============================================================
# 9. ADDITIONAL VECTORISED DATA OPERATION
# ============================================================

# Standardization using broadcasting
standardized_data = (data - mean_values) / std_values

print("\nSTANDARDIZED DATA:")
print(standardized_data[:5])


# ============================================================di
# 10. ASSERTIONS / VALIDATION
# ============================================================

assert array_1d.shape == (5,)
assert array_2d.shape == (2, 3)
assert array_3d.shape == (2, 2, 2)

assert data.shape[1] == 4
assert mean_values.shape == (4,)
assert std_values.shape == (4,)
assert correlation_matrix.shape == (4, 4)

print("\n" + "=" * 60)
print("ALL TASKS COMPLETED SUCCESSFULLY")
print("=" * 60)