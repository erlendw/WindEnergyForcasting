import CsvReader as read
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn.neural_network import MLPRegressor

weatherdata = read.getCsvAsList('TrainData')
forecastdata = read.getCsvAsList('WeatherForecastInput')
solutiondata = read.getCsvAsList('Solution')

x = list()
x2 = list()
y = list()
forecast_wind_speed = list()
forecast_wind_dir = list()
solution = list()

for i in range(len(weatherdata)):
    x.append(weatherdata[i][4])
    x2.append(float(weatherdata[i][2])+float(weatherdata[i][3]))
    y.append(weatherdata[i][1])

for i in range(len(forecastdata)):
    forecast_wind_speed.append(forecastdata[i][3])
    forecast_wind_dir.append(float(forecastdata[i][1])+float(forecastdata[i][2]))
    solution.append(solutiondata[i][1])

x = np.asarray(x,dtype=float)
x2 = np.array(x2,dtype=float)
y = np.asarray(y,dtype=float)
forecast_wind_speed = np.asarray(forecast_wind_speed,dtype=float)
forecast_wind_dir = np.asarray(forecast_wind_dir,dtype=float)


# Give data the correct dimensions 
x, x2, y, forecast,forecast2 = x.reshape(len(x),1), x2.reshape(len(x2),1), y.reshape(len(y), 1), forecast_wind_speed.reshape(len(forecast_wind_speed),1), forecast_wind_dir.reshape(len(forecast_wind_dir),1)

# Create array for mlr
x = np.hstack([x,x2])
forecast = np.hstack([forecast,forecast2])

# Train the Linear Regression Object
mlpr= MLPRegressor().fit(x,y)

# Predict
prediction = mlpr.predict(forecast)

# Calculate RMSE
sum_errors = 0
for i in range(len(prediction)):
	sum_errors += math.pow(2, (float(prediction[i])-float(solution[i])))

rmse = math.sqrt(sum_errors/len(prediction))

print(rmse)

# plot and show
plt.plot(range(len(solution)),solution)
plt.plot(range(len(prediction)),prediction, color='red')
plt.show()

