import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('Churn_Modelling.csv')
x=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values
#encodig Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x_1=LabelEncoder()
x[:,1]=labelencoder_x_1.fit_transform(x[:,1])
labelencoder_x_2=LabelEncoder()
x[:,2]=labelencoder_x_2.fit_transform(x[:,2])
onehotenc=OneHotEncoder(categorical_features=[1])
x=onehotenc.fit_transform(x).toarray()
x=x[:,1:]
#splitting dataset in train test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
#feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
#building ANN
import keras
from keras.models import Sequential
from keras.layers import Dense
#initalising the ANN
classifier=Sequential()
#adding input layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
#second layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
#output layer
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
#Compiling in ann
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#fit Ann to taining set
classifier.fit(x_train,y_train,nb_epoch=100)
#rpideiction
y_pred=classifier.predict(x_test)
y_pred=(y_pred>0.5)
#comfusion matics
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)



