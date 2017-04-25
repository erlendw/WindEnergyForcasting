from sklearn import neighbors
import CsvReader as read
import numpy as np
from pylab import scatter, show


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

clf = neighbors.KNeighborsClassifier()
#clf.fit(x_np, y_np)

print()

#print(clf.score(x_np, y_np))

scatter(x_np, y_np)
show()