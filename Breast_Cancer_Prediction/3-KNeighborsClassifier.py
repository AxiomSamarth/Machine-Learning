import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
from sklearn import neighbors, cross_validation
from sklearn import metrics
from matplotlib import style
style.use('fivethirtyeight')

df = pd.read_csv('BCW/breast-cancer-wisconsin.data.txt')
df.replace('?',-99999,inplace=True)
df.drop(['id'],1,inplace=True)
#print df

X = np.array(df.drop(['class'],1))
y = np.array(df['class'])

#print y
neighbors_i = []
accuracy_i = []

X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size = 0.2,random_state=4)

for i in range(1,31):
	knn = neighbors.KNeighborsClassifier(n_neighbors=i)
	knn.fit(X_train,y_train)

	accuracy = knn.score(X_test,y_test)
	neighbors_i.append(i)
	accuracy_i.append(accuracy)

#this plot gives us an idea of what must be the value of the number of neighbors to be considered for best prediction 
#and also to avoid over fitting of the curve
#use these plots to see how I choose the n_neighbors attribute

# plt.plot(neighbors_i,accuracy_i,color='black')
# plt.show()

best_number_of_neighbors = neighbors_i[accuracy_i.index(max(accuracy_i))]
knn = neighbors.KNeighborsClassifier(n_neighbors=best_number_of_neighbors)
knn.fit(X,y)
score = knn.score(X_test,y_test)
print 'The curve fitting accuracy is ',score*100,'%'

X_predict = np.array([[5,3,3,3,2,3,4,4,1]])
X_predict = X_predict.reshape(len(X_predict),-1)
y_predict = knn.predict(X_predict)

print 'The severity of the cancer for the given data is',

if y_predict == 2:
	print 'Benign'

elif y_predict == 4:
	print 'Malignant'	

else:
	print 'There was issue with the prediction. We don\'t prefer to go with a random one though. Sorry for imperfection!'