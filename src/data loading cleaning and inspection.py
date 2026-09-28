import numpy as np

# =====================================================================
# TASK 1: DATA LOADING & INITIAL INSPECTION
# =====================================================================

# Generating mock data: 10 rows (samples), 4 columns (features)
# In practice, replace with np.genfromtxt('dataset.csv', delimiter=',')
np.random.seed(42)
raw_data = np.random.uniform(low=10.0, high=100.0, size=(10, 4))

# Introduce missing/corrupted values (NaN) to simulate raw, dirty data
raw_data[2, 1] = np.nan
raw_data[5, 3] = np.nan

print("--- 1. Raw Data ---")
print(raw_data)
print("Shape:", raw_data.shape)
print("Data Type:", raw_data.dtype)

# Check for missing values
nan_count = np.isnan(raw_data).sum()
print(f"Total NaN values found: {nan_count}")


# =====================================================================
# TASK 2: DATA CLEANING (Handling NaNs using Column Means)
# =====================================================================

# Compute mean of each feature/column, ignoring NaNs
column_means = np.nanmean(raw_data, axis=0)
print("\n--- Column Means (Ignoring NaN) ---")
print(column_means)

# Replace NaNs with respective column means using boolean indexing
cleaned_data = raw_data.copy()
nan_indices = np.where(np.isnan(cleaned_data))

# Map column indices of NaNs to their respective mean values
cleaned_data[nan_indices] = column_means[nan_indices[1]]

print("\n--- Cleaned Data (NaNs Imputed) ---")
print(cleaned_data)


# =====================================================================
# TASK 3: ARRAY OPERATIONS, SLICING & BROADCASTING
# =====================================================================

# Slicing: Extract a subset (e.g., first 5 rows and features 0 & 2)
feature_subset = cleaned_data[:5, [0, 2]]
print("\n--- Sliced Subset (First 5 Rows, Cols 0 & 2) ---")
print(feature_subset)

# Broadcasting: Normalize dataset using Min-Max Scaling
# Formula: (X - X_min) / (X_max - X_min)
col_min = np.min(cleaned_data, axis=0)
col_max = np.max(cleaned_data, axis=0)

# Broadcasting subtracts col_min (1x4) and divides by range (1x4) across all rows (10x4)
normalized_data = (cleaned_data - col_min) / (col_max - col_min)

print("\n--- Normalized Data (Broadcasting) ---")
print(np.round(normalized_data, 4))


# =====================================================================
# TASK 4: MATHEMATICAL INSPECTION & FILTERING
# =====================================================================

# Statistical inspection across features
print("\n--- Data Inspection Summary ---")
print("Mean per feature:", np.round(np.mean(cleaned_data, axis=0), 2))
print("Std deviation per feature:", np.round(np.std(cleaned_data, axis=0), 2))
print("Overall Max value:", np.max(cleaned_data))

# Boolean Slicing: Filter rows where the first feature is above average
first_col_avg = np.mean(cleaned_data[:, 0])
high_val_rows = cleaned_data[cleaned_data[:, 0] > first_col_avg]

print(f"\n--- Rows where Feature 0 > Mean ({first_col_avg:.2f}) ---")
print(high_val_rows)