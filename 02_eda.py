import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv('data/city_day_clean.csv', parse_dates=['Date'])

print("Data loaded!")
print("Rows:", len(df))
print("Cities:", df['City'].unique())

os.makedirs('outputs', exist_ok=True)

city_avg = df.groupby('City')['AQI'].mean().sort_values(ascending=False)

plt.figure(figsize=(14, 6))
city_avg.plot(kind='bar', color='tomato', edgecolor='black')
plt.title('Average AQI by City', fontsize=16)
plt.xlabel('City')
plt.ylabel('Average AQI')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('outputs/chart1_city_aqi.png')
plt.show()

print("\nChart 1 done -- Average AQI by City")
print("Most polluted city:", city_avg.index[0], "| AQI:", round(city_avg.iloc[0], 1))
print("Cleanest city    :", city_avg.index[-1], "| AQI:", round(city_avg.iloc[-1], 1))

month_avg = df.groupby('Month')['AQI'].mean()
months = ['Jan','Feb','Mar','Apr','May','Jun',
          'Jul','Aug','Sep','Oct','Nov','Dec']

plt.figure(figsize=(10, 5))
plt.plot(months, month_avg.values, marker='o', color='crimson', linewidth=2.5)
plt.fill_between(months, month_avg.values, alpha=0.15, color='crimson')
plt.title('Average AQI by Month (Are winters worse?)', fontsize=15)
plt.xlabel('Month')
plt.ylabel('Average AQI')
plt.tight_layout()
plt.savefig('outputs/chart2_monthly_aqi.png')
plt.show()

worst_month_num = month_avg.idxmax()
print("\nChart 2 done -- Monthly Pattern")
print("Worst month:", months[worst_month_num - 1])
print("Best month :", months[month_avg.idxmin() - 1])


bucket_order  = ['Good','Satisfactory','Moderate','Poor','Very Poor','Severe']
bucket_colors = ['green','yellowgreen','orange','red','purple','black']

counts = df['AQI_Bucket'].value_counts().reindex(bucket_order, fill_value=0)

plt.figure(figsize=(10, 5))
plt.bar(counts.index, counts.values, color=bucket_colors, edgecolor='white')
plt.title('How Many Days in Each AQI Category?', fontsize=15)
plt.xlabel('AQI Category')
plt.ylabel('Number of Days')
plt.tight_layout()
plt.savefig('outputs/chart3_aqi_categories.png')
plt.show()

print("\nChart 3 done -- AQI Category Counts")
print(counts)

pivot = df.pivot_table(values='AQI', index='City', columns='Month', aggfunc='mean')
pivot.columns = months

plt.figure(figsize=(16, 8))
sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd',
            linewidths=0.5, linecolor='white')
plt.title('AQI by City and Month (darker = more polluted)', fontsize=15)
plt.tight_layout()
plt.savefig('outputs/chart4_heatmap.png')
plt.show()

print("\nChart 4 done -- Heatmap")

delhi = df[df['City'] == 'Delhi'].sort_values('Date')

plt.figure(figsize=(14, 5))
plt.plot(delhi['Date'], delhi['AQI'],
         color='tomato', linewidth=0.8, alpha=0.6, label='Daily AQI')

# Rolling average = smoothed line (30-day average)
delhi_smooth = delhi['AQI'].rolling(window=30).mean()
plt.plot(delhi['Date'], delhi_smooth,
         color='darkred', linewidth=2, label='30-day average')

plt.title("Delhi AQI Over Time", fontsize=15)
plt.xlabel('Date')
plt.ylabel('AQI')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/chart5_delhi_timeline.png')
plt.show()

print("\nChart 5 done -- Delhi Timeline")

sample = df[['PM2.5', 'AQI']].dropna().sample(2000, random_state=42)

plt.figure(figsize=(8, 5))
plt.scatter(sample['PM2.5'], sample['AQI'],
            alpha=0.3, color='steelblue', s=10)
plt.title('PM2.5 vs AQI -- Are They Related?', fontsize=14)
plt.xlabel('PM2.5')
plt.ylabel('AQI')
plt.tight_layout()
plt.savefig('outputs/chart6_pm25_vs_aqi.png')
plt.show()

print("\nChart 6 done -- PM2.5 vs AQI scatter plot")

print("\nALL 6 CHARTS DONE! Check your outputs/ folder.")