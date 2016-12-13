import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import neighbors, cross_validation, preprocessing
import numpy as np
from sklearn import metrics
style.use('fivethirtyeight')

df = pd.read_csv('glass.data.txt')
df.drop(['id'], 1, inplace=True)
# print df

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])
# print X

knn = neighbors.KNeighborsClassifier(n_neighbors=1)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.4,random_state=4)
knn.fit(X_train,y_train)

accuracy = knn.score(X_test,y_test)
print accuracy

y_predict = knn.predict(X_test)
accuracy = metrics.accuracy_score(y_test,y_predict)
print accuracy
