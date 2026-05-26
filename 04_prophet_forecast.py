import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data/city_day_clean.csv', parse_dates=['Date'])

delhi = df[df['City'] == 'Delhi'][['Date', 'AQI']].sort_values('Date')

delhi_monthly = delhi.set_index('Date').resample('MS').mean().dropna().reset_index()

delhi_monthly = delhi_monthly.rename(columns={'Date': 'ds', 'AQI': 'y'})

print("Data ready for Prophet:")
print(delhi_monthly.head())
print("\nTotal months:", len(delhi_monthly))

split = int(len(delhi_monthly) * 0.8)
train = delhi_monthly.iloc[:split]
test  = delhi_monthly.iloc[split:]

print("\nTraining on", len(train), "months, testing on", len(test), "months")

print("\nTraining Prophet model...")

model = Prophet(
    yearly_seasonality = True,
    weekly_seasonality = False,
    daily_seasonality  = False
)

model.fit(train)
print("Prophet model trained!")

future_for_test = model.make_future_dataframe(periods=len(test), freq='MS')
forecast_test = model.predict(future_for_test)

test_predictions = forecast_test[forecast_test['ds'].isin(test['ds'])]['yhat'].values

errors = abs(test['y'].values - test_predictions)
mae = errors.mean()

print("\nProphet Accuracy:")
print("On average, prediction was off by:", round(mae, 1), "AQI points")

full_model = Prophet(
    yearly_seasonality = True,
    weekly_seasonality = False,
    daily_seasonality  = False
)
full_model.fit(delhi_monthly)

future_dates = full_model.make_future_dataframe(periods=12, freq='MS')
forecast     = full_model.predict(future_dates)

next_12 = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12).copy()
next_12.columns = ['Month', 'Predicted AQI', 'Low Estimate', 'High Estimate']
next_12['Predicted AQI'] = next_12['Predicted AQI'].round(1)
next_12['Low Estimate']  = next_12['Low Estimate'].round(1)
next_12['High Estimate'] = next_12['High Estimate'].round(1)

print("\nProphet -- Next 12 Months Forecast:")
print(next_12.to_string(index=False))

plt.figure(figsize=(14, 5))
plt.plot(delhi_monthly['ds'], delhi_monthly['y'],
         label='Actual AQI', color='steelblue', linewidth=1.5)
plt.plot(forecast['ds'], forecast['yhat'],
         label='Prophet Forecast', color='darkorange', linewidth=2)
plt.fill_between(forecast['ds'],
                 forecast['yhat_lower'],
                 forecast['yhat_upper'],
                 alpha=0.2, color='orange',
                 label='95% confidence band')
plt.axvline(x=delhi_monthly['ds'].iloc[-1],
            color='grey', linestyle=':', label='Forecast starts here')
plt.title("Prophet Forecast -- Delhi AQI", fontsize=15)
plt.xlabel("Date")
plt.ylabel("AQI")
plt.legend()
plt.tight_layout()
plt.savefig('outputs/chart8_prophet_forecast.png')
plt.show()

print("\nChart 1 saved.")

seasonal = forecast[['ds', 'yearly']].copy()
seasonal['Month'] = seasonal['ds'].dt.month
monthly_pattern   = seasonal.groupby('Month')['yearly'].mean()

months = ['Jan','Feb','Mar','Apr','May','Jun',
          'Jul','Aug','Sep','Oct','Nov','Dec']

plt.figure(figsize=(10, 4))
plt.bar(months, monthly_pattern.values, color='steelblue', edgecolor='black')
plt.title("Seasonal Pattern -- Which Months Are Worse? (Prophet)", fontsize=14)
plt.xlabel("Month")
plt.ylabel("Seasonal Effect on AQI")
plt.axhline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig('outputs/chart9_prophet_seasonality.png')
plt.show()

print("Chart 2 saved -- Seasonal pattern")
print("\nPROPHET DONE! Charts saved to outputs/ folder")