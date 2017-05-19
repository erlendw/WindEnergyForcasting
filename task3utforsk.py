import CsvReader as read
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn.neural_network import MLPRegressor
import pyrenn
import pandas as pd

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

ny_x = []
ny_y = []
xpred = []
xpredbest = []
for i in range(14700,15000):
	add = []

	for k in range(i-30,i):
		add.append(y[k])

	#add.append(x[i-6])
	#add.append(x[i-5])
	#add.append(x[i-4])
	#add.append(x[i-3])
	#add.append(x[i-2])
	#add.append(x[i-1])
	#add.append(x[i])
	ny_x.append(add)
	ny_y.append(y[i+1])

for i in range(14701,len(y)-1):
	xpredbest.append(y[i])

ny_y.pop(len(ny_y)-1)


print(len(ny_x))

print(len(ny_y))
print(len(xpred))

x = ny_x

#y = x
#x.pop(len(x)-1)
#y.pop(0)




xpred = []
#xpred = [[x[len(x)-6], x[len(x)-5], x[len(x)-4]]]
xpred = x
xpred.pop(0)
xpred = np.asarray(xpred,dtype=float)
xpredbest = np.asarray(xpredbest,dtype=float)
x = np.asarray(x,dtype=float)
ny_y = np.asarray(ny_y,dtype=float)
forcast = np.asarray(forecast,dtype=float)

# Give data the correct dimensions 
#x, y, forecast = x.reshape(len(x),1), y.reshape(len(y), 1), forcast.reshape(len(forcast),1)
print(x.shape)
print(xpred.shape)
x = x.reshape(30,len(x))
xpred2 = []
xpred2.append(xpred[len(xpred)-3])
xpred2.append(xpred[len(xpred)-2])
xpred2.append(xpred[len(xpred)-1])
#xpred = xpred2
#xpred = np.asarray(xpred,dtype=float)
print(xpred)
print(xpred.shape)
xpred = xpred[0:len(xpred)/2]
xpred= xpred.reshape(30,len(xpred))

print(xpred.shape)

print(x.shape)
print(xpred)
#print(ny_y.shape)
# Train the Linear Regression Object
#mlpr= MLPRegressor().fit(x,y)
net = pyrenn.CreateNN([30,10,1])
#print(net)
net = pyrenn.train_LM(x,ny_y,net,verbose=True,k_max=200,E_stop=1e-2)


y2 = pyrenn.NNOut(xpred,net)



print(y2)

"""
ytest = pyrenn.NNOut(Ptest,net)
fig = plt.figure(figsize=(11,7))
ax0 = fig.add_subplot(211)
ax1 = fig.add_subplot(212)
fs=18

#Train Data
ax0.set_title('Train Data',fontsize=fs)
ax0.plot(x,y2,color='b',lw=2,label='NN Output')
ax0.plot(x,y,color='r',marker='None',linestyle=':',lw=3,markersize=8,label='Train Data')
ax0.tick_params(labelsize=fs-2)
ax0.legend(fontsize=fs-2,loc='upper left')
ax0.grid()

#Test Data
ax1.set_title('Test Data',fontsize=fs)
ax1.plot(Ptest,ytest,color='b',lw=2,label='NN Output')
ax1.plot(Ptest,Ytest,color='r',marker='None',linestyle=':',lw=3,markersize=8,label='Test Data')
ax1.tick_params(labelsize=fs-2)
ax1.legend(fontsize=fs-2,loc='upper left')
ax1.grid()

fig.tight_layout()
plt.show()
"""

# Predict
#prediction = mlpr.predict(forecast)

# Calculate RMSE
#sum_errors = 0
#for i in range(len(prediction)):
#	sum_errors += math.pow(2, (float(prediction[i])-float(solution[i])))

#rmse = math.sqrt(sum_errors/len(prediction))

#print(rmse)
print(xpredbest)
prediction = y2
 #plot and show
plt.plot(range(len(xpredbest)),xpredbest)
plt.plot(range(len(prediction)),prediction, color='red')
plt.show()

