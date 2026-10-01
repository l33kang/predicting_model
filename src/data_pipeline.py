import pandas as pd;
import numpy as np;
import hashlib
import json
import os


EXPECTED_COLUMNS = [
    "record_id",
    "plot_area_ha",
    "rainfall_mm",
    "soil_ph",
    "seed_kg",
    "distance_km",
    "arrival_hour",
    "actual_yield_kg",
    "dispatch_attention"
]

NUMERIC_COLUMNS = [
    "plot_area_ha",
    "rainfall_mm",
    "soil_ph",
    "seed_kg",
    "distance_km",
    "arrival_hour",
    "actual_yield_kg",
    "dispatch_attention"
]

FEATURE_COLUMNS = [
    "plot_area_ha",
    "rainfall_mm",
    "soil_ph",
    "seed_kg",
    "distance_km",
    "arrival_hour"
]

REGRESSION_TARGET = "actual_yield_kg"
CLASSIFICATION_TARGET = "dispatch_attention"

data_path = './data/dataSet.csv'

df = pd.read_csv(data_path)


print("\nDescriptive statistics:")
print(df[NUMERIC_COLUMNS].describe())

print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns)

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

actual_columns = list(df.columns)

if actual_columns != EXPECTED_COLUMNS:
    raise ValueError("Column names do not match expected columns.")

print("\nColumn validation: Passed");


for col in NUMERIC_COLUMNS:
    if not pd.api.types.is_numeric_dtype(df[col]):
        raise TypeError(f"Column '{col}' is not numeric.")
print("\nNumeric column validation: Passed");

X = df[FEATURE_COLUMNS]

y_regression = df[REGRESSION_TARGET]

y_classification = df[CLASSIFICATION_TARGET]

X_numpy = X.to_numpy()

print("\nFeature matrix (X):")
print(X_numpy)

print("\nFeature matrix (X) shape:")
print(X_numpy.shape)


# Calculate dataset SHA-256
with open(data_path, "rb") as file:
    dataset_hash = hashlib.sha256(file.read()).hexdigest()

print("\nDataset SHA-256:")
print(dataset_hash)


# Descriptive statistics
print("\nDescriptive statistics:")
print(df[NUMERIC_COLUMNS].describe())


# Create artifacts folder
os.makedirs("./artifacts", exist_ok=True)


# Create data report
report = {
    "row_count": len(df),
    "feature_count": len(FEATURE_COLUMNS),
    "missing_values": int(df.isnull().sum().sum()),
    "duplicate_rows": int(df.duplicated().sum()),
    "group_code": "AI-G2",
    "sha256": dataset_hash
}


# Save data report
with open("./artifacts/data_report.json", "w") as file:
    json.dump(report, file, indent=4)

print("\nData report saved to ./artifacts/data_report.json")