
from firebase import firebase


import numpy as np
from sklearn import datasets, neighbors, linear_model, tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import RandomizedPCA
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt
from time import time
from IPython import get_ipython
from sklearn.metrics import accuracy_score
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets,svm
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression


import pandas as pd
mydata = pd.read_csv('eeee.csv')

X = mydata.iloc[:,0:3].values
Y = mydata.iloc[:,3].values


number_of_samples = len(Y)

#Splitting into training and test sets
random_indices = np.random.permutation(number_of_samples)
#Training set
num_training_samples = int(number_of_samples*0.75)
x_train = X[random_indices[:num_training_samples]]
y_train = Y[random_indices[:num_training_samples]]
#Test set
x_test = X[random_indices[num_training_samples:]]
y_test = Y[random_indices[num_training_samples:]]

model = neighbors.KNeighborsClassifier(n_neighbors = 5) # K = 5
model.fit(x_train, y_train)
print(model)

y_expect=y_test
y_pred=model.predict(x_test)
print(metrics.classification_report(y_expect,y_pred))
print('Acc score KNN:',accuracy_score(y_test,y_pred))
z=confusion_matrix(y_test,y_pred)
print(z)

rf=RandomForestClassifier(n_estimators=100)
rf.fit(x_train,y_train)
print(rf)
y_expect1=y_test
y_pred1=rf.predict(x_test)
print(metrics.classification_report(y_expect1,y_pred1))
print('Acc score RandomForest:',accuracy_score(y_test,y_pred1))

z1=confusion_matrix(y_test,y_pred1)
print(z1)

svm=svm.SVC(kernel='linear',C=1,gamma=1)
svm.fit(x_train,y_train)
y_expect2=y_test
y_pred2=svm.predict(x_test)
print(metrics.classification_report(y_expect2,y_pred2))
print("Acc score SVM:",accuracy_score(y_test,y_pred2))
z2=confusion_matrix(y_test,y_pred2)
print(z2)


LogReg = LogisticRegression()
LogReg.fit(X,Y)
y_pred3=LogReg.predict(X)

print(classification_report(Y,y_pred3))
print("Acc score LogesticRegression:")
print (LogReg.score(X,Y))





firebase=firebase.FirebaseApplication('https://pulse-1e371.firebaseio.com/')
age=firebase.get('','age')
res=firebase.get('','value')
res1=firebase.get('','value2')
print("HeartBeat:")
print(res)
print("Exer Heartbeat:")
print(int(res1))





import csv

with open('test.csv','w',newline='') as f:
    thewriter=csv.writer(f)
    thewriter.writerow(['Age','Resting HR','Exer HR','StressLevel'])
    thewriter.writerow([age,res,res1,] )




myddata=pd.read_csv('test.csv')
X1 = myddata.iloc[:,0:3].values
Y1 = myddata.iloc[:,3].values

number_of_samples1 = len(Y1)
random_indices1 = np.random.permutation(number_of_samples1)

num_training_samples1 = int(number_of_samples1*0.99)
x1_test = X1[random_indices1[num_training_samples1:]]
y1_test = Y1[random_indices1[num_training_samples1:]]
tt=svm.predict(x1_test)
print("stress value:")
print(tt[0])
print(tt)
if(tt[0]==0):
    firebase.put('','stress','low')
if(tt[0]==1):
    firebase.put('','stress','medium')
if(tt[0]==2):
    firebase.put('','stress','high')
import csv
with open('test5.csv','a',newline='') as f:
    thewriter=csv.writer(f)
    
    thewriter.writerow([age,res,res1,tt] )










