from sklearn import neighbors
from sklearn.neighbors import KNeighborsRegressor
import CsvReader as read
import numpy as np
import matplotlib.pyplot as plt
import math

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
solution = np.asarray(solution,dtype=float)

# Give data the correct dimensions 
x, y, forecast= x.reshape(len(x),1), y.reshape(len(y), 1), forcast.reshape(len(forcast),1)

# Train 
knnReg = KNeighborsRegressor(10000, 'uniform').fit(x,y)

# Predict
prediction = knnReg.predict(forecast)

# Write to file
printlist = read.convert(prediction)
read.writeToFile("ForecastTemplate1-kNN.csv", dates, printlist)

# Calculate RMSE
rmse = np.sqrt(np.mean((solution-prediction)**2))

print(" ")
print("Prediction done using K-Nearest Neighbor")
print("RMSE: " + str(rmse))
print("Results stored in ForecastTemplate1-kNN.csv")

# plot and show
days = []
for i in range(1,32):
	days.append(str(i))
plt.xticks( np.arange(1,len(prediction)+2,24), days)
plt.xlabel("Dates")
plt.ylabel("Power")
plt.plot(range(len(solution)),solution)
plt.plot(range(len(prediction)),prediction, color='red')
plt.show()



