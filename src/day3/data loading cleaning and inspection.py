import numpy as np

np.random.seed(42)
raw_data = np.random.uniform(low=10.0, high=100.0, size=(10, 4))

raw_data[2, 1] = np.nan
raw_data[5, 3] = np.nan

print("--- 1. Raw Data ---")
print(raw_data)
print("Shape:", raw_data.shape)
print("Data Type:", raw_data.dtype)


nan_count = np.isnan(raw_data).sum()
print(f"Total NaN values found: {nan_count}")


column_means = np.nanmean(raw_data, axis=0)
print("\n--- Column Means (Ignoring NaN) ---")
print(column_means)

cleaned_data = raw_data.copy()
nan_indices = np.where(np.isnan(cleaned_data))

# Map column indices of NaNs to their respective mean values
cleaned_data[nan_indices] = column_means[nan_indices[1]]

print("\n--- Cleaned Data (NaNs Imputed) ---")
print(cleaned_data)

feature_subset = cleaned_data[:5, [0, 2]]
print("\n--- Sliced Subset (First 5 Rows, Cols 0 & 2) ---")
print(feature_subset)

col_min = np.min(cleaned_data, axis=0)
col_max = np.max(cleaned_data, axis=0)


normalized_data = (cleaned_data - col_min) / (col_max - col_min)

print("\n--- Normalized Data (Broadcasting) ---")
print(np.round(normalized_data, 4))

print("\n--- Data Inspection Summary ---")
print("Mean per feature:", np.round(np.mean(cleaned_data, axis=0), 2))
print("Std deviation per feature:", np.round(np.std(cleaned_data, axis=0), 2))
print("Overall Max value:", np.max(cleaned_data))


first_col_avg = np.mean(cleaned_data[:, 0])
high_val_rows = cleaned_data[cleaned_data[:, 0] > first_col_avg]

print(f"\n--- Rows where Feature 0 > Mean ({first_col_avg:.2f}) ---")
print(high_val_rows)