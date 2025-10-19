"""Original file is located at
    https://colab.research.google.com/drive/158CLUPLmS6XKopaLi9Cht0FeGJL42gIj
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

columns=['Sepal length','Sepal width','Petal length','Petal width','Class Labels']
df=pd.read_csv('iris.data.csv',names=columns, skiprows=1)
df.head()

df.describe()

sns.pairplot(df,hue='Class Labels')

data=df.values
X=data[:,0:4]
Y=data[:,4]
print(X)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

from sklearn.svm import SVC
model_svc=SVC()
model_svc.fit(X_train,Y_train)

prediction1=  model_svc.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test,prediction1)*100)

from sklearn.linear_model import LogisticRegression
model_lr=LogisticRegression()
model_lr.fit(X_train,Y_train)

prediction2=model_lr.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test,prediction2)*100)

from sklearn.tree import DecisionTreeClassifier
model_dt=DecisionTreeClassifier()
model_dt.fit(X_train,Y_train)

prediction3=model_dt.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test,prediction3)*100)

from sklearn.metrics import classification_report
print(classification_report(Y_test,prediction3))

X_new=np.array([[3,2,1,0.2],[4.9,2.2,3.8,1.1],[5.3,2.5,4.6,1.9]])
prediction=model_dt.predict(X_new)
print("Prediction of Species: {}".format(prediction))