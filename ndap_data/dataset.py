import pandas as pd

# Load your downloaded NDAP files
report_df = pd.read_csv("NDAP_REPORT_7234.csv")
keys_df = pd.read_csv("7234_KEYS.csv")

# Rename columns for clarity
ayush_df = report_df.rename(columns={
    'State': 'StateName',
    'Ayurveda pharmacies (UOM:Number) |Scaling Factor:1': 'Ayurveda',
    'Unani pharmacies (UOM:Number) |Scaling Factor:1': 'Unani',
    'Siddha pharmacies (UOM:Number) |Scaling Factor:1': 'Siddha',
    'Homoeopathy pharmacies (UOM:Number) |Scaling Factor:1': 'Homoeopathy'
})[['StateName', 'Ayurveda', 'Unani', 'Siddha', 'Homoeopathy']]

# Load demographics
pop_df = keys_df[['StateName', 'TotalPopulation', 'TotalPopulationUrban', 'TotalPopulationRural',
                  'LandArea', 'LandAreaUrban', 'LandAreaRural', 'NumberOfHouseholds']]

# Clean and merge
ayush_df['StateName'] = ayush_df['StateName'].str.lower().str.strip()
pop_df['StateName'] = pop_df['StateName'].str.lower().str.strip()
merged_df = pd.merge(ayush_df, pop_df, on='StateName', how='inner')

# Add derived columns
merged_df['Total_AYUSH'] = merged_df[['Ayurveda', 'Unani', 'Siddha', 'Homoeopathy']].sum(axis=1)
merged_df['AYUSH_per_lakh_population'] = merged_df['Total_AYUSH'] / (merged_df['TotalPopulation'] / 100000)

# Drop missing data rows
model_features = ['TotalPopulation', 'TotalPopulationUrban', 'TotalPopulationRural',
                  'LandArea', 'LandAreaUrban', 'LandAreaRural', 'NumberOfHouseholds', 'Total_AYUSH']
clean_df = merged_df.dropna(subset=model_features)

# Save to CSV
clean_df.to_csv("AYUSH_Merged_Cleaned.csv", index=False)
print("✅ AYUSH_Merged_Cleaned.csv saved successfully.")

