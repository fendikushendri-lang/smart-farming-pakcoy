import pandas as pd

excel_path = r"C:\Users\Asus\.gemini\antigravity\scratch\smart-farming-pakcoy-new\new Dataset_Pakcoy.xlsx"
df = pd.read_excel(excel_path)
temp_col = [c for c in df.columns if 'Temperature' in c][0]

print("=== CRITERIA DISTRIBUTION ===")
print(df['Ground Truth Criteria'].value_counts())

print("\n=== MATURITY LEVEL STATS BY CRITERIA ===")
for crit, group in df.groupby('Ground Truth Criteria'):
    print(f"\nCriteria: {crit}")
    print(f"Count: {len(group)} rows")
    print(f"Day range: {group['Day'].min()} - {group['Day'].max()}")
    print(f"DAP range: {group['DAP'].min()} - {group['DAP'].max()}")
    print(f"Maturity Level range: {group['Ground Truth Maturity Level (%)'].min()}% - {group['Ground Truth Maturity Level (%)'].max()}% (mean: {group['Ground Truth Maturity Level (%)'].mean():.2f}%)")
    print(f"Moisture range: {group['Soil Moisture (%)'].min()}% - {group['Soil Moisture (%)'].max()}%")
    print(f"Temp range: {group[temp_col].min()} - {group[temp_col].max()}")

print("\n=== CHECK FIRST 80 ROWS AND CRITERIA TRANSITIONS ===")
# Find rows where Criteria changes
df['Criteria_Change'] = df['Ground Truth Criteria'] != df['Ground Truth Criteria'].shift(1)
transitions = df[df['Criteria_Change']]
print(transitions[['Day', 'DAP', 'Time', 'Ground Truth Criteria', 'Ground Truth Maturity Level (%)']])

print("\n=== MATURITY LEVEL PROGRESSION OVER TIME ===")
print(df.groupby('DAP')[['Ground Truth Maturity Level (%)', 'Soil Moisture (%)', temp_col]].mean().head(30))
print("...")
print(df.groupby('DAP')[['Ground Truth Maturity Level (%)', 'Soil Moisture (%)', temp_col]].mean().tail(30))

print("\n=== MATURITY LEVEL UNIQUE VALUES ===")
print("Unique maturity levels:", sorted(df['Ground Truth Maturity Level (%)'].unique()))

print("\n=== UNIQUE VALUES OF TIME ===")
print(df['Time'].value_counts())
