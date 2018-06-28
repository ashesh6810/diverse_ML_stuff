import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt

geo = pd.read_csv('geo.csv', names = ["AC", "VTM", "Viscosity", "BD", "IDT", "Modulus", "Penetration"])

X_train=geo.drop(['Penetration'],axis=1)
y_train=geo['Penetration']

#X_train, X_test, y_train, y_test = train_test_split(X, y)

geo_test=pd.read_csv('geo_test.csv', names = ["AC", "VTM", "Viscosity", "BD", "IDT", "Modulus", "Penetration"])
X_test=geo_test.drop(['Penetration'],axis=1)
y_test=geo_test['Penetration']

scaler=StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPRegressor(random_state=35,max_iter=10000,hidden_layer_sizes=(300,))
mlp.fit(X_train,y_train)
predictions = mlp.predict(X_test)





fig=plt.figure(1)

plt.plot(predictions,'o',label='predicted value')
plt.plot(y_test,'*',label='actual value')
plt.xlabel('samples')
plt.ylabel('values')
plt.show()

E=np.zeros(8)
count=0
for j in range(5,45,5):
 mlp = MLPRegressor(random_state=j,max_iter=10000,hidden_layer_sizes=(300,))
 mlp.fit(X_train,y_train)
 predictions = mlp.predict(X_test)
 error=0
 for i in range(0,10):
  error=error+ pow((predictions[i]-y_test[i]),2)
  error=np.sqrt(error)/10
 E[count]=error
 count=count+1
fig=plt.figure(2)
plt.plot(E)

plt.show()
