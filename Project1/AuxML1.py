from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
import math

def normalizeNumberValues(testSet, trainSet, colsNormalize):
    for currentCol in testSet.columns:
        if currentCol in colsNormalize:
            testSet[currentCol] = (testSet[currentCol] - trainSet[currentCol].mean()) / trainSet[currentCol].std()
        else:
            continue
    return testSet


def hybridDistance(x, y, p, hybrid):
    if hybrid:
        distance = 0
        for index, value in x.items():
            if x[index].dtype == 'bool':
                distance += abs(int(x.loc[index]) - int(y.loc[index]))
            elif index == 'month':
                newDistance = abs(x.loc[index] - y.loc[index])
                if newDistance > 6:
                    newDistance = 12 - newDistance
                distance += newDistance
            elif index == 'day':
                newDistance = abs(x.loc[index] - y.loc[index])
                if newDistance > 3:
                    newDistance = 7 - newDistance
                distance += newDistance
            else:
                distance += abs(x.loc[index] - y.loc[index]) ** p
        return distance
    else:
        return (np.sum(abs(x - y) ** p)) ** (1 / p)


def kNearestNeighbor(targetRow, trainingData, validationSet, k, p, hybrid, error, kernel, isReg):
    # create our distance data frame to track all values
    distanceTable = pd.DataFrame({'distance': []})

    for comparisonRow in trainingData.index.tolist():
        # skip iteration when we are looking at same row
        if targetRow.name == comparisonRow:
            continue
        # find the difference between
        nextDistance = hybridDistance(targetRow, trainingData.loc[comparisonRow], p, hybrid)
        newRow = {'distance': nextDistance}
        distanceTable.loc[comparisonRow] = newRow

    # find the k smallest distances and join in their respective values
    kNearestTable = distanceTable.nsmallest(k, 'distance')
    kNearestTable = kNearestTable.join(validationSet)

    # find the class that is the most common occurance
    if isReg:
        estimate = math.exp(kernel * (sum(kNearestTable['Class'] * kNearestTable['distance']) / sum(kNearestTable['distance'])))
    else:
        estimate = kNearestTable['Class'].value_counts().idxmax()

    # create output series
    nearestNeighbor = {
        'testID': 0,
        'nearestNeighbors': kNearestTable.index.tolist(),
        'expectedValue': estimate,
        'actualValue': validationSet.loc[targetRow.name].Class
    }

    if isReg:
        valueDiff = abs(nearestNeighbor['expectedValue'] - nearestNeighbor['actualValue'])
        nearestNeighbor['correctAssignment'] = valueDiff < error
    else:
        nearestNeighbor['correctAssignment'] = nearestNeighbor['expectedValue'] == nearestNeighbor['actualValue']
    return nearestNeighbor


def condensedNearestNeighbor(originalDataSet, validationSet, k, p, hybrid, error, kernel, isReg):
    condensedSet = originalDataSet.sample()
    originalDataSet = originalDataSet.drop(condensedSet.index.tolist(), axis='index')
    condensedSetSize = len(condensedSet)
    updatedSize = 0

    while condensedSetSize != updatedSize:
        updatedSize = condensedSetSize
        for currentRow in originalDataSet.index.tolist():
            nextValueTest = kNearestNeighbor(originalDataSet.loc[currentRow], condensedSet, validationSet, k, p, hybrid, error, kernel, isReg)
            if not nextValueTest['correctAssignment']:
                condensedSet.loc[currentRow] = originalDataSet.loc[currentRow]
                originalDataSet = originalDataSet.drop(currentRow, axis='index')
        condensedSetSize = len(condensedSet)

    return condensedSet


def splitDataFrame(features, classes, splitPercentage):
    classes = classes.loc[features.index.tolist()]

    uniqueClassValues = (classes['Class'].unique().tolist())
    set1Classes = pd.DataFrame()
    set2Classes = pd.DataFrame()

    for nextClass in uniqueClassValues:
        subsetTable = classes[classes['Class'] == nextClass]
        subsetSize = round(len(subsetTable) * splitPercentage)
        set1Subset = subsetTable.sample(n=subsetSize)
        set1Classes = pd.concat([set1Classes, set1Subset])
        set2Indices = subsetTable.index.difference(set1Classes.index)
        set2Subset = subsetTable.loc[set2Indices]
        set2Classes = pd.concat([set2Classes, set2Subset])

    set1 = features.loc[set1Classes.index.tolist()]
    set2 = features.loc[set2Classes.index.tolist()]

    return set1, set2

def testEffectiveness(knnOutput, isReg):
    if isReg:
        return (1/len(knnOutput)) * sum((knnOutput['actualValue'] - knnOutput['expectedValue'])**2)
    else:
        return knnOutput['correctAssignment'].mean()
