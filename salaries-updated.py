import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

veriler=pd.read_csv("Salaries.csv")
raw=1000 #this file has 1488.655 rows so I need to decrease them
for column in list(veriler.columns):
    veriler.drop(veriler[veriler[column]=="Not provided"].index, inplace = True)
    #Some rows include missing value, we need to get out them

df=pd.concat([veriler.iloc[:,4:6],veriler.iloc[:,7:10]],axis=1)
x=df.iloc[0:raw,:]  #OvertimePay,OtherPay,TotalPay,TotalPayBenefits,Year
y=veriler.iloc[0:raw,2:3] #Predicting job title

X=x.values 
Y=y.values

from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.33)

#using standart scaler
from sklearn.preprocessing import StandardScaler 
sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)


#logistic regression
from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression(C=1168)
log_reg.fit(X_train,y_train)
y_pred=log_reg.predict(X_test)
score=accuracy_score(y_test,y_pred)
print("Logistic Regression")
print(score)

#k neighbours
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1,metric="chebyshev")
knn.fit(X_train,y_train)
y_pred2=knn.predict(X_test)
score=accuracy_score(y_test,y_pred2)
print("KNN")
print(score)

#svm
from sklearn.svm import SVC 
svc=SVC(kernel="rbf",degree=1,gamma=0.93,C=28.9) 
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)
score=accuracy_score(y_test,y_pred)
print("SVC")
print(score)

#naive bayes
from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(X_train,y_train)
y_pred=gnb.predict(X_test)
score=accuracy_score(y_test,y_pred)
print("Naive Bayes")
print(score)

#decission tree
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(criterion="entropy")
dtc.fit(X_train,y_train)
y_pred=dtc.predict(X_test)
score=accuracy_score(y_test,y_pred)
print("Decission Tree")
print(score)

#random forest
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=520,criterion="gini")
rfc.fit(X_train,y_train)
y_pred=rfc.predict(X_test)
score=accuracy_score(y_test,y_pred)
print("Random Forest")
print(score)

