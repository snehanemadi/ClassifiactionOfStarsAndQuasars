#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#python 3 implementation

import random
import csv

split = 0.80

#with open('cat3.csv') as csvfile:
    #lines = csv.reader(csvfile)
    #dataset = list(lines)
import pandas as pd
data=pd.read_csv('cat4.csv')
data=data.drop(['id','galex_objid','sdss_objid','spectrometric_redshift','pred'],axis=1)
from sklearn.utils import resample
class0= data[data['class']==0]#minority
class1= data[data['class']==1]#majority
class0_upsampled = resample(class0,
                          replace=True, # sample with replacement
                          n_samples=len(class1), # match number in majority class
                          random_state=123) # reproducible results
upsampled = pd.concat([class1, class0_upsampled])

#data.head()
dataset=upsampled.values.tolist()


random.shuffle(dataset)

div = int(split * len(dataset))
train = dataset [:div]
test = dataset [div:]
#print(train[0:5])
#print(test[0:5])


import math
# square root of the sum of the squared differences between the two arrays of numbers
def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        #print(instance1[x])
        #print(instance2[x])
        if(x!=11):
            #print(instance1[x])
            #print(instance2[x])
            distance += pow((float(instance1[x]) - float(instance2[x])), 2)
    return math.sqrt(distance)



import operator
#distances = []
def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def getResponse(neighbors):
    classVotes = {}
    classVotes['0']=0
    classVotes['1']=0
    for x in range(len(neighbors)):
        response = neighbors[x][11]
        #print(response)
        if response == 0.0:
            classVotes['0'] += 1
        else:
            classVotes['1'] += 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    #print(sortedVotes)
    return float(sortedVotes[0][0])

def confusion_matrix(act,pred):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    l0 = 0
    l1 = 0
    for i in range(len(act)):
        if(act[i]==0):
            l0 += 1
        elif(act[i]==1):
            l1 += 1
        if(act[i]==1 and pred[i]==1):
            tp += 1
        elif(act[i]==1 and pred[i]==0):
            fn += 1
        elif(act[i]==0 and pred[i]==1):
            fp += 1
        elif(act[i]==0 and pred[i]==0):
            tn += 1
    print("class 1 accuracy:",(tp/l1))
    print("class 0 accuracy:",(tn/l0))
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    recall = (tp)/(tp+fn)
    precision = (tp) / (tp+fp)
    a=2*(recall)*(precision)
    b=(recall+precision)
    f_score = a / b
    error_rate = 1-accuracy
    print("accuracy:" + repr(accuracy))
    print("recall:" + repr(recall))
    print("precision:" + repr(precision))
    print("f-score:" + repr(f_score))
    print("error rate:" + repr(error_rate))

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        #print(predictions[x])
        #print(testSet[x][14])
        if testSet[x][11] == float(predictions[x]):
            correct += 1
    return (correct/float(len(testSet))) * 100.0

predictions=[]

k = 7

for x in range(len(test)):
    #print(len(test[x]))
    neighbors = getNeighbors(train, test[x], k)
    #print("N",neighbors)
    result = getResponse(neighbors)
    #print("R",result)
    predictions.append(result)
    #print(predictions)
    #print('> predicted=' + repr(result) + ', actual=' + repr(test[x][11]))

accuracy = getAccuracy(test, predictions)
print('Accuracy: ' + repr(accuracy) + '%')
actual = []
for i in range(len(test)):
    actual.append(test[i][11])

results = confusion_matrix(actual, predictions) 

