import pandas as pd
import numpy as np

excel_path = r"C:\Users\Asus\.gemini\antigravity\scratch\smart-farming-pakcoy-new\new Dataset_Pakcoy.xlsx"
df = pd.read_excel(excel_path)
temp_col = [c for c in df.columns if 'Temperature' in c][0]

print("=== CHECK FIRST 50 ROWS ===")
print(df.head(20).to_string())

print("\n=== CHECK ROWS AROUND DAY 20-35 ===")
print(df[(df['Day'] >= 20) & (df['Day'] <= 30)].head(30).to_string())

print("\n=== CHECK ROWS AROUND DAY 100-110 ===")
print(df[(df['Day'] >= 100) & (df['Day'] <= 103)].to_string())

print("\n=== CHECK ROWS AT END (DAY 375-380) ===")
print(df.tail(20).to_string())

print("\n=== SOIL CONDITION DISTRIBUTION ===")
print(df['Soil Condition'].value_counts())
print("\nRows with Dry soil:")
print(df[df['Soil Condition'] == 'Dry'])
print("\nRows with Wet soil:")
print(df[df['Soil Condition'] == 'Wet'])

print("\n=== CORRELATION MATRIX ===")
numeric_cols = ['Day', 'DAP', 'Soil Moisture (%)', temp_col, 'Ground Truth Maturity Level (%)']
print(df[numeric_cols].corr())

print("\n=== CHECK DUPLICATES ===")
print("Exact duplicate rows:", df.duplicated().sum())
print("Duplicate (Day, Time):", df.duplicated(subset=['Day', 'Time']).sum())
