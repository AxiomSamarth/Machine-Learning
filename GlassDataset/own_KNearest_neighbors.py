import numpy as np
import pandas as pd
import random
from collections import Counter
from math import *
import matplotlib.pyplot as plt

df = pd.read_csv('glass.data.txt')
df.drop(['id'], 1, inplace=True)
full_data = df.astype(float).values.tolist()
# print full_data

def k_nearest_neighbors(data,predict,k=1):

    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance,group])

    votes = [i[1] for i in sorted(distances)][:k]
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result
accuracies = []
for iteration in range(30):
    random.shuffle(full_data)
    test_size = 0.4
    test_set = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[]}
    train_set = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[]}
    test_data = full_data[-int(test_size*len(full_data)):]
    train_data = full_data[:-int(test_size*len(full_data))]

    for i in test_data:
        test_set[i[-1]].append(i[:-1])

    for i in train_data:
        train_set[i[-1]].append(i[:-1])

    correct = 0
    total = 0

    for group in test_set:
        for i in test_set[group]:
            vote = k_nearest_neighbors(train_set,i)
            if group == vote:
                correct += 1
            total += 1

    accuracies.append(correct*100/total)
    # print "The accuracy of the K nearest neighbors implemented from scratch is",(correct*100/total)

print "The average accuracy percentage of the algorithm being run is",ceil(np.mean(np.array(accuracies)))