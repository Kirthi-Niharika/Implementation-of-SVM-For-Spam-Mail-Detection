# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages.
2. Import the dataset to operate on.
3. Split the dataset.
4. Predict the required output.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: T Kirthi Niharika
RegisterNumber: 212221040084
*/
import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result
import pandas as pd
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:
![image](https://github.com/Kirthi-Niharika/Implementation-of-SVM-For-Spam-Mail-Detection/assets/114135005/ff738267-b938-4439-9d46-9b93745d7962)
## data.head():
![image](https://github.com/Kirthi-Niharika/Implementation-of-SVM-For-Spam-Mail-Detection/assets/114135005/4effd531-8cb4-494a-9b32-d1c8458b6891)
## data.info():
![image](https://github.com/Kirthi-Niharika/Implementation-of-SVM-For-Spam-Mail-Detection/assets/114135005/b9c17303-1210-4f53-b325-0797f5b67086)
## data.isnull().sum():
![image](https://github.com/Kirthi-Niharika/Implementation-of-SVM-For-Spam-Mail-Detection/assets/114135005/cc5baa0c-afbb-4e6f-af35-88c98c26116f)
## Y_prediction value:
![image](https://github.com/Kirthi-Niharika/Implementation-of-SVM-For-Spam-Mail-Detection/assets/114135005/1afe9c03-5ca0-44f1-b431-6475df11bed7)
 ## Accuracy value:
![image](https://github.com/Kirthi-Niharika/Implementation-of-SVM-For-Spam-Mail-Detection/assets/114135005/c67e92fd-3ef0-49aa-848e-4158a2a62eda)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
