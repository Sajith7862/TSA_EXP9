# EX.NO.09        A project on Time series analysis on weather forecasting using ARIMA model 
### Date: 

### NAME: MOHAMED HAMEEM SAJITH J
### REG NO : 212223240090

### AIM:
To Create a project on Time series analysis on weather forecasting using ARIMA model in  Python and compare with other models.
### ALGORITHM:
1. Explore the dataset of weather 
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions
### PROGRAM:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

try:
    df_full = pd.read_csv('GlobalLandTemperaturesByCity.csv')
except FileNotFoundError:
    print("Error: 'GlobalLandTemperaturesByCity.csv' not found.")
    print("Please make sure the file is in the same directory as your script.")
else:
    df_city = df_full[df_full['City'] == 'Ahmadabad'].copy()

    df_city['Date'] = pd.to_datetime(df_city['dt'])

    df_city = df_city.set_index('Date')

    df = df_city[['AverageTemperature']].dropna()

    data_monthly = df['AverageTemperature'].resample('MS').mean().ffill()

    data = data_monthly.to_frame(name='AverageTemperature')

    print("First 5 rows of the filtered dataset:")
    print(data.head())
    print("\n")

    if data.empty:
        print("Error: Processed data is empty. No valid temperature data found for the selected city after filtering and cleaning.")
    else:
        def arima_model(data, target_variable, order):
            
            train_size = int(len(data) * 0.8)
            train_data, test_data = data[:train_size], data[train_size:]

            print(f"Training data size: {len(train_data)}")
            print(f"Testing data size: {len(test_data)}\n")

            if train_data.empty:
                 print("Error: Training data is empty. Cannot fit ARIMA model.")
                 return

            print("Fitting ARIMA model... (This may take a moment)")
            model = ARIMA(train_data[target_variable], order=order)
            fitted_model = model.fit()
            print("Model fitting complete.\n")

            forecast = fitted_model.forecast(steps=len(test_data))

            rmse = np.sqrt(mean_squared_error(test_data[target_variable], forecast))
            print("--- EVALUATION ---")
            print(f"Root Mean Squared Error (RMSE): {rmse:.4f}\n")

            print("Displaying Forecast Plot...")
            plt.figure(figsize=(12, 6))
            plt.plot(train_data[target_variable].iloc[-200:], label='Training Data (Last 200 pts)')
            plt.plot(test_data.index, test_data[target_variable], label='Testing Data (Actual)', color='orange')
            plt.plot(test_data.index, forecast, label='Forecasted Data', color='green', linestyle='--')
            plt.xlabel('Date')
            plt.ylabel(target_variable)
            plt.title(f'ARIMA {order} Forecasting for {target_variable}')
            plt.legend()
            plt.grid(True)
            plt.show()

        arima_model(data, 'AverageTemperature', order=(5,1,0))
```
### OUTPUT:

<img width="1010" height="547" alt="image" src="https://github.com/user-attachments/assets/2354c835-694f-4dae-9492-246b1d6155b2" />

### RESULT:
Thus the program run successfully based on the ARIMA model using python.
