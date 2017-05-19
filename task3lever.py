import CsvReader as read
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn.neural_network import MLPRegressor
import pyrenn
import pandas as pd
import copy

weatherdata = read.getCsvAsList('TrainData')
forcastdata = read.getCsvAsList('WeatherForecastInput')
solutiondata = read.getCsvAsList('Solution')

x = list()
y = list()
x1 = list()
y1 = list()
forecast = list()
solution = list()
x = []
y = []
xpred = []
xpredbest = []

for i in range(len(weatherdata)):
    x1.append(weatherdata[i][4])
    y1.append(weatherdata[i][1])

for i in range(len(forcastdata)):
    forecast.append(forcastdata[i][3])
    solution.append(solutiondata[i][1])

top = len(y1)

# arranging input data
for i in range(top-301,top-1):
	add = []

	for k in range(i-30,i):
		add.append(y1[k])

	x.append(add)
	y.append(y1[i+1])

for i in range(top-300,top-1):
	xpredbest.append(y1[i])

y.pop(len(y)-1)
x.pop(0)
inputdata = []
inputdata = copy.deepcopy(x)


#inputdata = inputdata[len(inputdata)-1]
check = copy.deepcopy(inputdata)


inputdata = np.asarray(inputdata,dtype=float)
xpredbest = np.asarray(xpredbest,dtype=float)
x = np.asarray(x,dtype=float)
y = np.asarray(y,dtype=float)

x = x.reshape(30,len(x))
inputdata = inputdata.reshape(30,len(inputdata))

#Create NN
net = pyrenn.CreateNN([30,10,1])

# Train
net = pyrenn.train_LM(x,y,net,verbose=True,k_max=200,E_stop=1e-2)

#Predict
prevout = pyrenn.NNOut(inputdata,net)

 #plot and show
plt.plot(range(len(xpredbest)),xpredbest)
plt.plot(range(len(prevout)),prevout, color='red')
plt.show()

