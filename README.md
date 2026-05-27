# 🌫️ AQI Analysis & Forecasting -- Indian Cities
 
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Status](https://img.shields.io/badge/Status-Completed-green)
 
## 📌 Overview
A complete data science project analyzing historical Air Quality Index (AQI)
data across 26 major Indian cities and forecasting future pollution levels
using ARIMA and Facebook Prophet models.
 
## 🎯 Objective
- Analyze 5 years of daily AQI data (2015-2020)
- Uncover seasonal patterns and city-wise pollution trends
- Forecast AQI for the next 12 months
- Provide actionable insights for policymakers
 
## 📂 Project Structure
```
aqi_project/
│
├── data/
│   └── city_day_clean.csv
│
├── outputs/
│   └── (10 charts saved here)
│
├── 01_data_cleaning.py
├── 02_eda.py
├── 03_arima_forecast.py
├── 04_prophet_forecast.py
└── 05_insights.py
```
 
## 📊 Dataset
- **Source:** Kaggle -- Air Quality Data in India by Rohan Rao
- **Original Source:** Central Pollution Control Board (CPCB)
- **Size:** ~29,531 rows x 16 columns
- **Cities:** 26 Indian cities including Delhi, Mumbai, Kolkata, Bengaluru
 
## 🛠️ Libraries Used
| Library | Purpose |
|---------|---------|
| pandas | Data loading and manipulation |
| numpy | Numerical computations |
| matplotlib | Data visualization |
| seaborn | Statistical charts and heatmaps |
| statsmodels | ARIMA forecasting model |
| prophet | Facebook Prophet forecasting |
| scikit-learn | Model evaluation metrics |
 
## 📈 Key Findings
- Delhi had the highest average AQI -- most polluted major city in India
- November and December were the worst months due to winter and Diwali
- Aizawl had the lowest average AQI -- cleanest city in the dataset
- PM2.5 had a correlation above 0.90 with AQI -- confirmed as top pollutant
- Prophet outperformed ARIMA in capturing seasonal patterns
 
## 🔮 Models Used
| Model | Type | Use |
|-------|------|-----|
| ARIMA (2,1,2) | Statistical | Time series forecasting |
| Facebook Prophet | ML-based | Seasonal forecasting with confidence bands |
 
## 💡 Policy Recommendations
1. Odd-even vehicle scheme in Delhi from November to February
2. Public health alerts 3 days before and after Diwali
3. Study green zone planning of Aizawl and Shillong
4. Stricter night-time industrial emission rules
5. Expand CPCB monitoring to Tier-2 cities
 
## 🚀 How to Run
```bash
pip install pandas numpy matplotlib seaborn statsmodels prophet scikit-learn
 
python 01_data_cleaning.py
python 02_eda.py
python 03_arima_forecast.py
python 04_prophet_forecast.py
python 05_insights.py
```
