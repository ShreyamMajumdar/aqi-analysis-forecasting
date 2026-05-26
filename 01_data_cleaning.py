import pandas as pd
import os

print("Loading data...")
df = pd.read_csv("data/city_day.csv")

print("Done! Here are the first 5 rows:")
print(df.head())

print("\nShape of data (rows, columns):", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nHow many missing values in each column:")
print(df.isnull().sum())

df['Date'] = pd.to_datetime(df['Date'])

df['Month'] = df['Date'].dt.month
df['Year']  = df['Date'].dt.year

print("\nDate column fixed. Month and Year columns added.")

number_columns = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']

for col in number_columns:
    if col in df.columns:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)

print("Missing values filled with median.")

df = df[(df['AQI'] >= 0) & (df['AQI'] <= 500)]

print("Removed rows with AQI outside 0-500 range.")

def get_bucket(aqi):
    if aqi <= 50:    return 'Good'
    elif aqi <= 100: return 'Satisfactory'
    elif aqi <= 200: return 'Moderate'
    elif aqi <= 300: return 'Poor'
    elif aqi <= 400: return 'Very Poor'
    else:            return 'Severe'

df['AQI_Bucket'] = df['AQI'].apply(get_bucket)

os.makedirs('data', exist_ok=True)
df.to_csv('data/city_day_clean.csv', index=False)

print("\nDONE! Clean file saved to: data/city_day_clean.csv")
print("Total rows:", len(df))
print("Cities in data:", df['City'].nunique())
print("\nAll cities:")
print(df['City'].unique())