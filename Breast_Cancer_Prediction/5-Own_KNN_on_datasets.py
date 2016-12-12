import numpy as np
import pandas as pd
from collections import Counter
import warnings
from math import sqrt
import random

accuracies = []

for i in range(25):
    df = pd.read_csv("BCW/breast-cancer-wisconsin.data.txt")
    df.replace('?',-99999,inplace=True)
    df.drop(['id'],1,inplace=True)
    full_data = df.astype(float).values.tolist()
    random.shuffle(full_data)

    def k_nearest_neighbors(data,predict,k=5):

        distances = []
        for group in data:
            for features in data[group]:
                euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
                distances.append([euclidean_distance,group])

        votes = [i[1] for i in sorted(distances)[:k]]
        vote_result = Counter(votes).most_common(1)[0][0]
        confidence = Counter(votes).most_common(1)[0][1]/k

        return vote_result,confidence


    test_size = 0.2
    train_set = {2:[],4:[]}
    test_set = {2:[],4:[]}
    train_data = full_data[:-int(test_size*len(full_data))]
    test_data = full_data[-int(test_size*len(full_data)):]


    for i in test_data:
        test_set[i[-1]].append(i[:-1])

    for i in train_data:
        train_set[i[-1]].append(i[:-1])

    correct = 0
    total = 0

    for group in test_set:
        for i in test_set[group]:
            vote,confidence = k_nearest_neighbors(train_set,i)
            if group == vote:
                correct += 1
            total += 1

    accuracies.append(correct*100/total)

print 'The average accuracy is',(sum(accuracies)/len(accuracies)),'%'