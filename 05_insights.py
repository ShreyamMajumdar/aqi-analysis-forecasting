import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data/city_day_clean.csv', parse_dates=['Date'])

print("=" * 55)
print("  FINAL REPORT -- AQI Analysis of Indian Cities")
print("=" * 55)

city_avg = df.groupby('City')['AQI'].mean().sort_values(ascending=False)

print("\nTOP 5 MOST POLLUTED CITIES:")
for i, (city, aqi) in enumerate(city_avg.head(5).items(), 1):
    print(" ", i, ".", city, "-- Average AQI:", round(aqi, 1))

print("\nTOP 5 CLEANEST CITIES:")
cleanest = city_avg.tail(5).sort_values(ascending=True)
for i, (city, aqi) in enumerate(cleanest.items(), 1):
    print(" ", i, ".", city, "-- Average AQI:", round(aqi, 1))

month_avg  = df.groupby('Month')['AQI'].mean()
months_map = {
    1:'January',  2:'February', 3:'March',    4:'April',
    5:'May',      6:'June',     7:'July',      8:'August',
    9:'September',10:'October', 11:'November', 12:'December'
}

worst_month = months_map[month_avg.idxmax()]
best_month  = months_map[month_avg.idxmin()]

print("\nWORST MONTH :", worst_month, "(Avg AQI:", round(month_avg.max(), 1), ")")
print("BEST MONTH  :", best_month,  "(Avg AQI:", round(month_avg.min(), 1), ")")

severe         = df[df['AQI_Bucket'].isin(['Severe', 'Very Poor'])]
severe_by_city = severe.groupby('City').size().sort_values(ascending=False)

print("\nCITIES WITH MOST SEVERE / VERY POOR DAYS:")
for city, days in severe_by_city.head(5).items():
    print(" ", city, ":", days, "days")

if 'PM2.5' in df.columns:
    corr = df['PM2.5'].corr(df['AQI'])
    print("\nPM2.5 correlation with AQI:", round(corr, 2))
    print("(1.0 = perfect -- PM2.5 is the #1 driver of AQI)")

print("\nRunning both models to compare accuracy...")
print("(this takes about 30 seconds -- please wait)")

delhi = df[df['City'] == 'Delhi'][['Date', 'AQI']].sort_values('Date')
delhi_monthly = delhi.set_index('Date').resample('MS').mean().dropna()

split = int(len(delhi_monthly) * 0.8)
train_data = delhi_monthly.iloc[:split]
test_data  = delhi_monthly.iloc[split:]

arima_model = ARIMA(train_data['AQI'], order=(2, 1, 2)).fit()
arima_preds = arima_model.forecast(steps=len(test_data))
arima_mae   = abs(test_data['AQI'].values - arima_preds.values).mean()

prophet_train = train_data.reset_index().rename(
    columns={'Date': 'ds', 'AQI': 'y'}
)
prophet_model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False
)
prophet_model.fit(prophet_train)

future       = prophet_model.make_future_dataframe(periods=len(test_data), freq='MS')
prophet_fc   = prophet_model.predict(future)
test_dates   = test_data.index
prophet_preds = prophet_fc[prophet_fc['ds'].isin(test_dates)]['yhat'].values
prophet_mae  = abs(test_data['AQI'].values - prophet_preds).mean()

print("\nMODEL COMPARISON:")
print("  ARIMA   -- Average error:", round(arima_mae, 1), "AQI points")
print("  Prophet -- Average error:", round(prophet_mae, 1), "AQI points")

if prophet_mae < arima_mae:
    print("  RESULT: Prophet performed BETTER for this dataset")
else:
    print("  RESULT: ARIMA performed BETTER for this dataset")

months = ['Jan','Feb','Mar','Apr','May','Jun',
          'Jul','Aug','Sep','Oct','Nov','Dec']

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('AQI Analysis -- Final Summary Dashboard',
             fontsize=17, fontweight='bold')

top10 = city_avg.head(10)
axes[0, 0].barh(top10.index[::-1], top10.values[::-1], color='tomato')
axes[0, 0].set_title('Top 10 Most Polluted Cities')
axes[0, 0].set_xlabel('Average AQI')

axes[0, 1].plot(months, month_avg.values,
                marker='o', color='darkorange', linewidth=2)
axes[0, 1].set_title('Monthly AQI Pattern')
axes[0, 1].set_ylabel('Average AQI')

bucket_order  = ['Good','Satisfactory','Moderate','Poor','Very Poor','Severe']
bucket_colors = ['green','yellowgreen','orange','red','purple','black']
counts = df['AQI_Bucket'].value_counts().reindex(bucket_order, fill_value=0)
axes[1, 0].pie(counts.values, labels=counts.index,
               colors=bucket_colors, autopct='%1.1f%%', startangle=140)
axes[1, 0].set_title('Overall AQI Category Split')

axes[1, 1].bar(['ARIMA', 'Prophet'], [arima_mae, prophet_mae],
               color=['steelblue', 'darkorange'], edgecolor='black', width=0.4)
axes[1, 1].set_title('ARIMA vs Prophet -- Prediction Error')
axes[1, 1].set_ylabel('Mean Absolute Error (lower = better)')
for i, val in enumerate([arima_mae, prophet_mae]):
    axes[1, 1].text(i, val + 0.5, str(round(val, 1)),
                    ha='center', fontsize=13)

plt.tight_layout()
plt.savefig('outputs/chart10_summary_dashboard.png', dpi=150)
plt.show()

print("\nSummary dashboard saved to outputs/chart10_summary_dashboard.png")

print("\n" + "=" * 55)
print("  POLICY RECOMMENDATIONS (use these in your report)")
print("=" * 55)
print("""
1. VEHICLE CONTROL
   Delhi and Kanpur have AQI above 300 in winter.
   Suggestion: Odd-even vehicle scheme from Nov to Feb.

2. DIWALI WARNING SYSTEM
   AQI spikes sharply around Diwali every year.
   Suggestion: Public health alerts 3 days before Diwali.

3. LEARN FROM CLEAN CITIES
   Some cities consistently stay below AQI 80.
   Suggestion: Study their green zones and traffic systems.

4. FACTORY EMISSION CHECKS
   SO2 and NO2 spikes point to industrial pollution.
   Suggestion: Stricter night-time factory emission rules.

5. MORE MONITORING STATIONS
   Data gaps show many cities are not being tracked.
   Suggestion: Expand CPCB monitoring to Tier-2 cities.
""")

print("=" * 55)
print("  PROJECT COMPLETE! All outputs saved in outputs/")
print("=" * 55)