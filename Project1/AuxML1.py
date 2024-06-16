from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd

def normalizeNumberValues(testSet, trainSet, colsNormalize):
    """
    Used to normalize values within data set where relevant

    @param testSet: the test set to be normalized
    @param trainSet: the train set for which the mean and std will be sourced from
    @param colsNormalize: the columns for which normalizing is needed within the test set
    @return: normalized dataframe
    """

    # iterate through our columns
    for currentCol in testSet.columns:
        # normalize the column if specified
        if currentCol in colsNormalize:
            testSet[currentCol] = (testSet[currentCol] - trainSet[currentCol].mean()) / trainSet[currentCol].std()
        else:
            continue
    return testSet


def hybridDistance(x, y, p, hybrid):
    """
    Used to determine distance between data points. Handles numeric, boolean, and two unique
    cases for days of week and months of year.

    @param x: data point 1
    @param y: data point 2
    @param p: the exponent of the Minkowski formula
    @param hybrid: whether all of the data is numeric or not.
    @return: distance between data points
    """

    # if hybrid, more complex calc
    if hybrid:
        # initialize distance
        distance = 0
        # iterate through data sets
        for index, value in x.items():
            # if boolean values, cast each as int and take diff
            if x[index].dtype == 'bool':
                distance += abs(int(x.loc[index]) - int(y.loc[index]))
            # if month, find true distance knowing it cant be more than 6 months
            elif index == 'month':
                newDistance = abs(x.loc[index] - y.loc[index])
                if newDistance > 6:
                    newDistance = 12 - newDistance
                distance += newDistance
            # if day, find true distance knowing it cant be more than 3 days
            elif index == 'day':
                newDistance = abs(x.loc[index] - y.loc[index])
                if newDistance > 3:
                    newDistance = 7 - newDistance
                distance += newDistance
            else:
                distance += abs(x.loc[index] - y.loc[index]) ** p
        return distance

    # if not hybrid, the distance is a simple subtraction of data sets, then take Minkowski metric based on p provided
    else:
        return (np.sum(abs(x - y) ** p)) ** (1 / p)


def kNearestNeighbor(targetRow, trainingData, validationSet, k, p, hybrid, error, kernel, isReg):
    """
    Determines the k nearest data points to the point provided given a training set to compare against.

    @param targetRow: the row for which we are trying to find neighbors for
    @param trainingData: the data set for which we will find the neighbors from
    @param validationSet: the set of relevant targets for our data set
    @param k: the number of neighbors to consider
    @param p: the exponent of the Minkowski formula
    @param hybrid: whether all features are numeric or not
    @param error: the range for which we will accept regression estimates
    @param kernel: the multiplication factor within our kernal factor
    @param isReg: whether or not the estimate is a regression or not
    @return: a dataframe about the neighbors chosen and the estimate generated was successful
    """
    # create our distance data frame to track all values
    distanceTable = pd.DataFrame({'distance': []})

    # iterate through training set
    for comparisonRow in trainingData.index.tolist():
        # skip iteration when we are looking at same row
        if targetRow.name == comparisonRow:
            continue
        # find the difference between the target value and the current row of the training set
        nextDistance = hybridDistance(targetRow, trainingData.loc[comparisonRow], p, hybrid)
        newRow = {'distance': nextDistance}
        distanceTable.loc[comparisonRow] = newRow

    # find the k smallest distances and join in their respective values
    kNearestTable = distanceTable.nsmallest(k, 'distance')
    kNearestTable = kNearestTable.join(validationSet)

    # estimate the output of the target value
    # estimate if regression
    if isReg:
        relevantTargets = validationSet.loc[trainingData.index.tolist()]
        # if only 1 value, take that value
        if len(relevantTargets['Class'].unique()) == 1:
            estimate = relevantTargets['Class'].unique()[0]
        # if more than 1 value, take estimate using our kernal smoother
        else:
            testSetStd = relevantTargets['Class'].std()
            kNearestTable['kernelEstimate'] = np.exp((1/(kernel * testSetStd) * kNearestTable['distance']))
            kNearestTable['weightedEstimate'] = kNearestTable['kernelEstimate'] * kNearestTable['Class']
            estimate = sum(kNearestTable['weightedEstimate']) / sum(kNearestTable['kernelEstimate'])
    # estimate if class
    else:
        # take the most common value of the neighbors
        estimate = kNearestTable['Class'].value_counts().idxmax()

    # create output series
    nearestNeighbor = {
        'testID': 0,
        'nearestNeighbors': kNearestTable.index.tolist(),
        'expectedValue': estimate,
        'actualValue': validationSet.loc[targetRow.name].Class
    }

    # check for correct regression
    # if regression, test output against the error term included
    if isReg:
        valueDiff = abs(nearestNeighbor['expectedValue'] - nearestNeighbor['actualValue'])
        nearestNeighbor['correctAssignment'] = valueDiff < error
    # otherwise, test if output matches actual value
    else:
        nearestNeighbor['correctAssignment'] = nearestNeighbor['expectedValue'] == nearestNeighbor['actualValue']
    return nearestNeighbor


def condensedNearestNeighbor(originalDataSet, validationSet, k, p, hybrid, error, kernel, isReg):
    """
    Used to condensing down set of values that we will used to test our values against to improve testing performance

    @param originalDataSet: the set to be condensed
    @param validationSet: the set of relevant targets for our data set
    @param k: the number of neighbors to consider
    @param p: the exponent of the Minkowski formula
    @param hybrid: whether all features are numeric or not
    @param error: the range for which we will accept regression estimates
    @param kernel: the multiplication factor within our kernal factor
    @param isReg: whether or not the estimate is a regression or not
    @return: a dataframe of the condensed test set
    """

    # start with 1 random sample to begin condensed set
    condensedSet = originalDataSet.sample()
    # remove that sample from the larger set
    originalDataSet = originalDataSet.drop(condensedSet.index.tolist(), axis='index')
    # intialize our size values to track condensed set as we go
    condensedSetSize = len(condensedSet)
    updatedSize = 0

    # run until condensed set size does not change
    while condensedSetSize != updatedSize:
        # reset size
        updatedSize = condensedSetSize
        # iterate through our remaining test set
        for currentRow in originalDataSet.index.tolist():
            # check for nearest neighbors, note that we always pass in 1 for k here
            nextValueTest = kNearestNeighbor(originalDataSet.loc[currentRow], condensedSet, validationSet, k, p, hybrid, error, kernel, isReg)
            # if we don't get the correct value as defined in above function, add to our condensed set
            if not nextValueTest['correctAssignment']:
                condensedSet.loc[currentRow] = originalDataSet.loc[currentRow]
                originalDataSet = originalDataSet.drop(currentRow, axis='index')
        # reset our condensed set size
        condensedSetSize = len(condensedSet)

    return condensedSet


def splitDataFrame(features, classes, splitPercentage, isReg):
    """
    Used to stratify and split our data sets where needed to ensure an equal distribution of the data set

    @param features: the data set to be split
    @param classes: the relevant target values for our
    @param splitPercentage: the percent for which to cut the dataset into
    @param isReg: whether the data set is a regression or not
    @return: two dataframes, the first split into the size of the splitPercentage, the other with remainder
    """

    # subset the targets by the features inputted
    classes = classes.loc[features.index.tolist()]

    # initialize our outputs
    set1Classes = pd.DataFrame()
    set2Classes = pd.DataFrame()

    # if regression, we do not need to worry about stratification
    if isReg:
        # copy over the set of targets and find size
        subsetTable = classes.copy()
        subsetSize = round(len(subsetTable) * splitPercentage)

        # take a random sample of the subset with size as defined above, add to output
        set1Subset = subsetTable.sample(n=subsetSize)
        set1Classes = pd.concat([set1Classes, set1Subset])

        # find what indices were not added to first set and add to second
        set2Indices = subsetTable.index.difference(set1Classes.index)
        set2Subset = subsetTable.loc[set2Indices]
        set2Classes = pd.concat([set2Classes, set2Subset])

    # otherwise, stratify by target output
    else:
        # find all unique outputs
        uniqueClassValues = (classes['Class'].unique().tolist())

        # iterate through each class
        for nextClass in uniqueClassValues:
            # subset by current class and find size to split off
            subsetTable = classes[classes['Class'] == nextClass]
            subsetSize = round(len(subsetTable) * splitPercentage)

            # sample from the class subset by size specified above and add to output
            set1Subset = subsetTable.sample(n=subsetSize)
            set1Classes = pd.concat([set1Classes, set1Subset])

            # find the indices not added previously and add to the second data set
            set2Indices = subsetTable.index.difference(set1Classes.index)
            set2Subset = subsetTable.loc[set2Indices]
            set2Classes = pd.concat([set2Classes, set2Subset])

    # return the features based on our split class sets
    set1 = features.loc[set1Classes.index.tolist()]
    set2 = features.loc[set2Classes.index.tolist()]

    return set1, set2

def testEffectiveness(knnOutput, isReg):
    """
    Tests the effectiveness of the algorithm once it has run

    @param knnOutput: the final dataset
    @param isReg: whether the dataset is a regression or not
    @return: numeric value that represents effectiveness of test. Categorical returns a % correct, regression returns an MSE
    """

    # if regression, return MSE
    if isReg:
        return np.mean((knnOutput['actualValue'] - knnOutput['expectedValue'])**2)
    # otherwise, return % correct
    else:
        return knnOutput['correctAssignment'].mean()
