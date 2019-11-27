#!/usr/bin/env python
# coding: utf-8

# In[4]:


import random
import csv
from random import randrange
import pandas as pd
import math
import operator
from sklearn.utils import resample

#cross-validating cat3.csv with five folds 
data=pd.read_csv('cat3.csv')
data=data.drop(['id','spectrometric_redshift','pred'],axis=1)

#Upsampling stars since cat3.csv has less instances of stars in comparison of quesars
#class 0 indicates stars and class 1 indicates quesars

class0= data[data['class']==0]
class1= data[data['class']==1]
class0_upsampled = resample(class0,replace=True,n_samples=len(class1),random_state=123) 
upsampled = pd.concat([class1, class0_upsampled])
dataset=upsampled.values.tolist()
train = dataset

#function to calculate distance, ignoring the column "15" since it is the class(i.e to be predicted) column in the dataset 
def euclideanDistance(i1, i2, length):
    dist = 0
    for i in range(length):
        if(i!=14):
            dist += pow((float(i1[i]) - float(i2[i])), 2)
    return math.sqrt(dist)

#function to get the nearest neighbours
def neighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    n = []
    for x in range(k):
        n.append(distances[x][0])
    return n

  

def votes(n):
    neigh_votes = {}
    neigh_votes['0']=0
    neigh_votes['1']=0
    for x in range(len(n)):
        response = neighbors[x][14]

        if response == 0.0:
            neigh_votes['0'] += 1
        else:
            neigh_votes['1'] += 1
    sortedVotes = sorted(neigh_votes.items(), key=operator.itemgetter(1), reverse=True)

    return float(sortedVotes[0][0])


def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][14] == float(predictions[x]):
                correct += 1
    return (correct/float(len(testSet))) * 100.0

def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = dataset
    fold_size = int(len(dataset_copy) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

def knn_algo(train,test,k):
    predictions=[]
    for x in range(len(test)):

        neighbors = neighbors(train, test[x], k)

        result = votes(neighbors)
        predictions.append(result)

    return predictions
        
    

def evaluate_algorithm(dataset, algorithm, n_folds, k):
    dataset_split = cross_validation_split(dataset, n_folds)
    scores = list()
    for cur_fold in dataset_split:
        train=list()
        test_set = cur_fold
        train_set = dataset_split.copy()
 
        ind = train_set.index(cur_fold)
        del(train_set[ind])
        
        for fold in train_set:
            for row in fold:
                train.append(row)
        
        
        predictions = algorithm(train, test_set, k)

        
        accuracy = getAccuracy(test_set, predictions)
        scores.append(accuracy)
      
    return scores



k = 7
n_folds = 5

scores = evaluate_algorithm(train, knn_algo, n_folds, k)

print('Scores: %s' % scores)
for i in range(len(scores)):
    print("Each Fold accuracy:"+repr(scores[i]))
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))


# In[ ]:




