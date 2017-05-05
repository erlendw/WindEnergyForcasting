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

y = x
x.pop(len(x)-1)
y.pop(0)

x = np.asarray(x,dtype=float)
y = np.asarray(y,dtype=float)
forcast = np.asarray(forecast,dtype=float)

# Give data the correct dimensions 
#x, y, forecast = x.reshape(len(x),1), y.reshape(len(y), 1), forcast.reshape(len(forcast),1)

print(x.shape)
print(y.shape)
df = pd.ExcelFile('example_data.xlsx').parse('friction')
#x = df.loc[0:40]['P']
#y = df.loc[0:40]['Y']
Ptest = df['Ptest'].values
Ytest = df['Ytest'].values

print(x.shape)
print(y.shape)
# Train the Linear Regression Object
#mlpr= MLPRegressor().fit(x,y)
net = pyrenn.CreateNN([1,10,1])
net = pyrenn.train_LM(x,y,net,verbose=True,k_max=10,E_stop=1e-5)
y2 = pyrenn.NNOut(x,net)
print(y2)
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
# Predict
#prediction = mlpr.predict(forecast)

# Calculate RMSE
#sum_errors = 0
#for i in range(len(prediction)):
#	sum_errors += math.pow(2, (float(prediction[i])-float(solution[i])))

#rmse = math.sqrt(sum_errors/len(prediction))

#print(rmse)

# plot and show
#plt.plot(range(len(solution)),solution)
#plt.plot(range(len(prediction)),prediction, color='red')
#plt.show()

