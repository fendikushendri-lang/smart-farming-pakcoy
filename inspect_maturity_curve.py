import pandas as pd
import numpy as np

df = pd.read_excel(r'C:\Users\Asus\.gemini\antigravity\scratch\smart-farming-pakcoy-new\new Dataset_Pakcoy.xlsx')
temp_col = [c for c in df.columns if 'Temperature' in c][0]

daily = df.groupby('Day').agg({
    'DAP': 'first',
    'Soil Moisture (%)': ['mean', 'min', 'max'],
    temp_col: ['mean', 'min', 'max'],
    'Ground Truth Maturity Level (%)': 'first',
    'Ground Truth Criteria': 'first'
})

print("First 35 days (daily):")
print(daily.head(35).to_string())

print("\nDays 36 to 70 (daily sample):")
print(daily.iloc[35:70:5].to_string())

print("\nDays 300 to 380 (daily sample):")
print(daily.iloc[300::10].to_string())

print("\nRate of increase of Maturity Level:")
mat = daily[('Ground Truth Maturity Level (%)', 'first')]
diffs = mat.diff()
print("Value counts of daily maturity increase:")
print(diffs.value_counts())
