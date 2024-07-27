import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

np.random.seed(123)
data = np.cumsum(np.random.randn(100))  
model = ARIMA(data, order=(1, 1, 1))

model_fit = model.fit()


print(model_fit.summary())

residuals = model_fit.resid
plt.figure(figsize=(10, 6))
plt.subplot(211)
plt.plot(residuals)
plt.title('Residuals')
plt.subplot(212)
plt.hist(residuals, bins=30)
plt.title('Histogram of Residuals')
plt.tight_layout()
plt.show()

# Forecast future values
forecast_steps = 10  # Number of steps to forecast
forecast = model_fit.forecast(steps=forecast_steps)

# Plot the forecast
plt.figure(figsize=(10, 6))
plt.plot(data, label='Observed')
plt.plot(range(len(data), len(data) + forecast_steps), forecast, label='Forecast', color='red')
plt.legend()
plt.show()
