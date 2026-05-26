import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data/city_day_clean.csv', parse_dates=['Date'])

delhi = df[df['City'] == 'Delhi'][['Date', 'AQI']].sort_values('Date')

delhi_monthly = delhi.set_index('Date').resample('MS').mean().dropna()

print("Delhi monthly AQI data (first 10 rows):")
print(delhi_monthly.head(10))
print("\nTotal months of data:", len(delhi_monthly))

split = int(len(delhi_monthly) * 0.8)
train = delhi_monthly.iloc[:split]
test  = delhi_monthly.iloc[split:]

print("\nTraining months:", len(train))
print("Testing months :", len(test))

print("\nTraining ARIMA model... (takes a few seconds)")

model = ARIMA(train['AQI'], order=(2, 1, 2))
model_fit = model.fit()

print("Model trained!")

predictions = model_fit.forecast(steps=len(test))

errors = abs(test['AQI'].values - predictions.values)
mae    = errors.mean()

print("\nModel Accuracy:")
print("On average, prediction was off by:", round(mae, 1), "AQI points")
print("(Lower = better)")

full_model = ARIMA(delhi_monthly['AQI'], order=(2, 1, 2))
full_model_fit = full_model.fit()

future_forecast = full_model_fit.forecast(steps=12)

last_date = delhi_monthly.index[-1]
future_dates = pd.date_range(
    start=last_date + pd.DateOffset(months=1),
    periods=12,
    freq='MS'
)

print("\nForecasted AQI for next 12 months:")
for date, aqi in zip(future_dates, future_forecast.values):
    print(" ", date.strftime('%b %Y'), "-->", round(aqi, 0))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

ax1.plot(train.index, train['AQI'],
         label='Training Data', color='steelblue')
ax1.plot(test.index, test['AQI'],
         label='Actual AQI', color='green', linewidth=2)
ax1.plot(test.index, predictions,
         label='ARIMA Prediction', color='red',
         linewidth=2, linestyle='--')
ax1.set_title('ARIMA: Prediction vs Actual (Delhi)', fontsize=14)
ax1.set_ylabel('AQI')
ax1.legend()

ax2.plot(delhi_monthly.index, delhi_monthly['AQI'],
         label='Historical AQI', color='steelblue')
ax2.plot(future_dates, future_forecast.values,
         label='12-Month Forecast', color='darkorange',
         linewidth=2.5, marker='o')
ax2.axvline(x=delhi_monthly.index[-1],
            color='grey', linestyle=':', label='Forecast starts here')
ax2.set_title('ARIMA: Next 12 Months Forecast (Delhi)', fontsize=14)
ax2.set_ylabel('AQI')
ax2.legend()

plt.tight_layout()
plt.savefig('outputs/chart7_arima.png')
plt.show()

print("\nARIMA DONE! Chart saved to outputs/chart7_arima.png")