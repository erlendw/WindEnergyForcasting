from sklearn import neighbors
from sklearn.neighbors import KNeighborsRegressor
import CsvReader as read
import numpy as np
from pylab import scatter, show
import matplotlib.pyplot as plt
import math

weatherdata = read.getCsvAsList('TrainData')
forcastdata = read.getCsvAsList('WeatherForecastInput')
solutiondata = read.getCsvAsList('Solution')

x = list()
y = list()
forecast = list()
solution = list()

for i in range(len(weatherdata)):
    x.append(weatherdata[i][4])
    y.append(weatherdata[i][1])

for i in range(len(forcastdata)):
    forecast.append(forcastdata[i][3])
    solution.append(solutiondata[i][1])

x = np.asarray(x,dtype=float)
y = np.asarray(y,dtype=float)
forcast = np.asarray(forecast,dtype=float)

# Give data the correct dimensions 
x, y, forecast= x.reshape(len(x),1), y.reshape(len(y), 1), forcast.reshape(len(forcast),1)

# Train 
knnReg = KNeighborsRegressor(10, 'uniform').fit(x,y)
prediction = knnReg.predict(forecast)

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



