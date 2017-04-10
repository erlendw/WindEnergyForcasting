import CsvReader as read
import numpy as np
from numpy import genfromtxt
from pylab import scatter, show, title, xlabel, ylabel, plot, contour
from sklearn.linear_model import LinearRegression

weatherdata = read.getCsvAsList('TrainData')
forestdata = read.getCsvAsList('WeatherForecastInput')

watherList = list()
powerOutPutList = list()
forecast = list()

for i in range(len(weatherdata)):
    watherList.append(weatherdata[i][4])
    powerOutPutList.append(weatherdata[i][1])

for i in range(len(forestdata)):
    forecast.append(forestdata[i][3])

x_np = np.asarray(watherList,dtype=float)
y_np = np.asarray(powerOutPutList,dtype=float)
forcast_np = np.asarray(forecast,dtype=float)

slope, intercept = np.polyfit(x_np, y_np, 1)

output = slope * forcast_np + intercept


scatter(x_np, y_np)
plot(x_np,slope * x_np + intercept, color="red")
#scatter(forcast_np, output, color="blue")

#print(regr.coef_)

x, y = x_np.reshape(len(x_np),1), y_np.reshape(len(y_np), 1)

x_pre = forcast_np.reshape(len(forcast_np),1)

print(x.shape)
print(y.shape)

# Linear Regression Object
lin_regression = LinearRegression()

# Fitting linear model to the data
lin_regression.fit(x,y)

# Get slope of fitted line
m = lin_regression.coef_

# Get y-Intercept of the Line
b = lin_regression.intercept_

predictions = np.asarray(lin_regression.predict(x_pre))
predictions2 = slope * x_pre + intercept



for i in range(len(predictions)):
    print(predictions[i], output[i])

#plot(x_np,m * x_np + b, color="pink")
show()
