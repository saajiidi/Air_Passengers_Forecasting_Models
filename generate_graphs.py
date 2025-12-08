import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns
import os

# Ensure images directory exists
if not os.path.exists('images'):
    os.makedirs('images')

print("Loading data...")
# Load data
df = pd.read_csv('AirPassengers.csv', parse_dates=['Month'], index_col='Month')
# Ensure frequency is set
df.index.freq = 'MS'

train = df.iloc[:int(len(df)*0.8)]
test = df.iloc[int(len(df)*0.8):]

print("Training models...")

# 1. Holt-Winters
hw_model = ExponentialSmoothing(train, seasonal='mul', trend='add', seasonal_periods=12).fit()
hw_forecast = hw_model.forecast(len(test))

# 2. SARIMA
# Using order from APF.py: order=(1,1,1), seasonal_order=(1,1,1,12)
sarima_model = ARIMA(train, order=(1,1,1), seasonal_order=(1,1,1,12)).fit()
sarima_forecast = sarima_model.forecast(len(test))

# 3. ARIMA
# Using order from APF.py: (5,1,0)
arima_model = ARIMA(train, order=(5,1,0)).fit()
arima_forecast = arima_model.forecast(len(test))

models = {
    "Holt-Winters": hw_forecast,
    "SARIMA": sarima_forecast,
    "ARIMA": arima_forecast
}

# Evaluation
print("Evaluating...")
results = []
for name, forecast in models.items():
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mae = mean_absolute_error(test, forecast)
    mape = np.mean(np.abs((test.values.flatten() - forecast.values.flatten()) / test.values.flatten())) * 100
    results.append({"Model": name, "RMSE": rmse, "MAE": mae, "MAPE": mape})

results_df = pd.DataFrame(results)

# PLOT 1: Forecast vs Actual
print("Generating forecast plot...")
plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label='Training Data', color='gray', alpha=0.5)
plt.plot(test.index, test, label='Actual Test Data', color='black', linewidth=2)
for name, forecast in models.items():
    plt.plot(test.index, forecast, label=name, linestyle='--')

plt.title("Forecast vs Actual (Top Models)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('images/forecast_vs_actual.png', dpi=300, bbox_inches='tight')
plt.close()

# PLOT 2: Model Metrics
print("Generating metrics plot...")
plot_df = results_df.melt(id_vars="Model", value_vars=["RMSE", "MAE", "MAPE"], 
                         var_name="Metric", value_name="Value")

plt.figure(figsize=(10, 6))
# Set style if possible, wrapped in try-except just in case
try:
    sns.set_palette("viridis")
except:
    pass

sns.barplot(data=plot_df, x="Model", y="Value", hue="Metric")
plt.title("Model Comparison by Metrics")
plt.grid(axis='y', alpha=0.3)
plt.savefig('images/model_metrics.png', dpi=300, bbox_inches='tight')
plt.close()

print("Graphs generated successfully.")
