# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
``
```
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.
```
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: sreenithi.E
RegisterNumber:212223220109  
*/
```
```

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('student.csv')
df.head(10)
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
x=df.iloc[:,0:-1]
y=df.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,Y_train)
X_train
Y_train
lr.predict(x_test.iloc[0].values.reshape(1,1))
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(X_train,lr.predict(X_train),color='orange')
lr.coef_
lr.intercept_

```



## Output:

![Screenshot 2024-05-07 155532](https://github.com/sreenithi23/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147017600/94f67475-ed6a-447d-95aa-b0c54a16d0cf)

![Screenshot 2024-05-07 155550](https://github.com/sreenithi23/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147017600/8c79e18d-dc8e-4f95-bb87-ad3d698f8aa8)

![Screenshot 2024-05-07 155603](https://github.com/sreenithi23/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147017600/c241a0eb-7ab2-48ce-abe0-127d87cbf75c)

![Screenshot 2024-05-07 155615](https://github.com/sreenithi23/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147017600/29c59d75-e42f-4d89-99fc-b29550c9330a)

![Screenshot 2024-05-07 155626](https://github.com/sreenithi23/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147017600/2014935a-f326-4337-92e5-991f0672c05f)

![Screenshot 2024-05-07 155632](https://github.com/sreenithi23/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147017600/0f078bcc-878c-4335-a080-af6a12fc0b50)






## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
