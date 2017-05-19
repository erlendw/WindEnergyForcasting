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

for i in range(len(weatherdata)):
    x1.append(weatherdata[i][4])
    y1.append(weatherdata[i][1])

for i in range(len(forcastdata)):
    forecast.append(forcastdata[i][3])
    solution.append(solutiondata[i][1])

top = len(y1)
x = []
y = []
xpred = []
xpredbest = []
"""
for i in range(14700,15000):
	add = []

	for k in range(i-30,i):
		add.append(y1[k])

	x.append(add)
	y.append(y1[i+1])

for i in range(14701,len(y1)-1):
	xpredbest.append(y1[i])

y.pop(len(y)-1)

"""
for i in range(top-301,top-1):
	add = []

	for k in range(i-30,i):
		add.append(y1[k])

	#add.append(x[i-6])
	#add.append(x[i-5])
	#add.append(x[i-4])
	#add.append(x[i-3])
	#add.append(x[i-2])
	#add.append(x[i-1])
	#add.append(x[i])
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
print(x.shape)
print(y.shape)
print(inputdata.shape)


net = pyrenn.CreateNN([30,10,1])
net = pyrenn.train_LM(x,y,net,verbose=True,k_max=200,E_stop=1e-2)


prevxpred = xpred
prevout = pyrenn.NNOut(inputdata,net)
plt.plot(range(len(prevout)),prevout, color='red')

previnputdata = inputdata
temp = inputdata.tolist()
temp1 = prevout.tolist()
temp.append(temp1[len(temp1)-31:len(temp1)-1])

print(temp)
print(len(temp[len(temp)-1]))
print(len(temp[len(temp)-2]))
inputdata = np.asarray(temp,dtype=float)
for i in range(30):

	current = pyrenn.NNOut(inputdata,net,[previnputdata,prevout])
	prevout = current
	previnputdata = inputdata
	plt.plot(range(len(current)),current, color='green')
	temp = inputdata.tolist()
	temp.append(current[len(current)-31:len(current)-1])
	inputdata = np.asarray(temp,dtype=float)

	#new = xpred[]
	print("halla")
	print(y2)


print(prevout)
print(xpredbest)
prediction = prevout


 #plot and show
plt.plot(range(len(xpredbest)),xpredbest)

plt.show()

