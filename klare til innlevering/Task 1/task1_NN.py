import CsvReader as read
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn.neural_network import MLPRegressor

weatherdata = read.getCsvAsList('TrainData')
forcastdata = read.getCsvAsList('WeatherForecastInput')
solutiondata = read.getCsvAsList('Solution')

x = list()
y = list()
forecast = list()
solution = list()
dates = list()

# Fill lists
for i in range(len(weatherdata)):
    x.append(weatherdata[i][4])
    y.append(weatherdata[i][1])

for i in range(len(forcastdata)):
    forecast.append(forcastdata[i][3])
    solution.append(solutiondata[i][1])
    dates.append(solutiondata[i][0])

# Create Numpy array
x = np.asarray(x,dtype=float)
y = np.asarray(y,dtype=float)
forcast = np.asarray(forecast,dtype=float)
solution = np.asarray(solution,dtype=float)

# Give data the correct dimensions 
x, y, forecast = x.reshape(len(x),1), y.reshape(len(y), 1), forcast.reshape(len(forcast),1)

# Train the Linear Regression Object
mlpr= MLPRegressor().fit(x,y.ravel())

# Predict
prediction = mlpr.predict(forecast)

# Write to file
read.writeToFile("ForecastTemplate1-NN.csv", dates, prediction)

# Calculate RMSE
rmse = np.sqrt(np.mean((solution-prediction)**2))

print(" ")
print("Prediction done using Neural Network")
print("RMSE: " + str(rmse))
print("Results stored in ForecastTemplate1-NN.csv")


# plot and show
plt.plot(range(len(solution)),solution)
plt.plot(range(len(prediction)),prediction, color='red')
plt.show()

