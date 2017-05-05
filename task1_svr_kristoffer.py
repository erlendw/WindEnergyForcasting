import CsvReader as read
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn import svm

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

# Create numpy arrays
x = np.asarray(x,dtype=float)
y = np.asarray(y,dtype=float)
forcast = np.asarray(forecast,dtype=float)

# Give data the correct dimensions 
x, y, forecast = x.reshape(len(x),1), y.reshape(len(y), 1), forcast.reshape(len(forcast),1)

# Train the Linear Regression Object
svc= svm.SVR().fit(x,y)

# Predict
prediction = svc.predict(forecast)

# Write to file
read.writeToFile("ForecastTemplate1-SVR.csv", dates, prediction)

# Calculate RMSE
sum_errors = 0
for i in range(len(prediction)):
	sum_errors += math.pow(2, (float(prediction[i])-float(solution[i])))

rmse = math.sqrt(sum_errors/len(prediction))

print(" ")
print("Prediction done using Supported Vector Regression")
print("RMSE: " + str(rmse))
print("Results stored in ForecastTemplate1-SVR.csv")

# plot and show
plt.plot(range(len(solution)),solution)
plt.plot(range(len(prediction)),prediction, color='red')
plt.show()

