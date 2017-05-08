import CsvReader as read
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn.linear_model import LinearRegression

weatherdata = read.getCsvAsList('TrainData')
forecastdata = read.getCsvAsList('WeatherForecastInput')
solutiondata = read.getCsvAsList('Solution')

x = list()
x2 = list()
y = list()
forecast_wind_speed = list()
forecast_wind_dir = list()
solution = list()
dates = list()

for i in range(len(weatherdata)):
    x.append(weatherdata[i][4])
    x2.append(float(weatherdata[i][2])+float(weatherdata[i][3]))
    y.append(weatherdata[i][1])

for i in range(len(forecastdata)):
    forecast_wind_speed.append(forecastdata[i][3])
    forecast_wind_dir.append(float(forecastdata[i][1])+float(forecastdata[i][2]))
    solution.append(solutiondata[i][1])
    dates.append(solutiondata[i][0])
    #print(solutiondata[i][0])

x = np.asarray(x,dtype=float)
x2 = np.array(x2,dtype=float)
y = np.asarray(y,dtype=float)
forecast_wind_speed = np.asarray(forecast_wind_speed,dtype=float)
forecast_wind_dir = np.asarray(forecast_wind_dir,dtype=float)
solution = np.asarray(solution,dtype=float)


# Give data the correct dimensions 
x, x2, y, forecast, forecast2 = x.reshape(len(x),1), x2.reshape(len(x2),1), y.reshape(len(y), 1), forecast_wind_speed.reshape(len(forecast_wind_speed),1), forecast_wind_dir.reshape(len(forecast_wind_dir),1)


# Train the Linear Regression Object
lin_reg= LinearRegression().fit(x,y)

# Predict
prediction_LR = lin_reg.predict(forecast)



# Create array for mlr
x = np.hstack([x,x2])
forecast = np.hstack([forecast,forecast2])

# Train the Multiple Linear Regression Object
mlpr= LinearRegression().fit(x,y)

# Predict
prediction_MLR = mlpr.predict(forecast)



# Calculate RMSE
rmse_LR = np.sqrt(np.mean((solution-prediction_LR)**2))

rmse_MLR = np.sqrt(np.mean((solution-prediction_MLR)**2))

# Write to file
read.writeToFile("ForecastTemplate2.csv", dates, prediction_MLR)

print(" ")
print("Prediction done using Multiple Linear Regression")
print("RMSE for Multiple Linear Regression: " + str(rmse_MLR))
print("RMSE for Linear Regression: " + str(rmse_LR))
print("Results stored in ForecastTemplate2.csv")

# plot and show
days = []
for i in range(1,32):
	days.append(str(i))
plt.xticks( np.arange(1,len(prediction_LR)+2,24), days)
plt.xlabel("Dates")
plt.ylabel("Power")
plt.plot(range(len(solution)),solution)
plt.plot(range(len(prediction_LR)),prediction_LR, color='red')
plt.plot(range(len(prediction_MLR)),prediction_MLR, color='green')
plt.show()

