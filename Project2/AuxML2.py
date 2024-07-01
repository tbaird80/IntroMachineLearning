import pandas as pd
import numpy as np
import copy
import random
import TreeClass

def splitDataFrame(dataSet, splitPercentage, isReg):
    """
    Used to stratify and split our data sets where needed to ensure an equal distribution of the data set

    @param dataSet: the data set to be split
    @param splitPercentage: the percent for which to cut the dataset into
    @param isReg: whether the data set is a regression or not
    @return: two dataframes, the first split into the size of the splitPercentage, the other with remainder
    """
    # initialize our outputs
    set1 = pd.DataFrame()
    set2 = pd.DataFrame()

    # if regression, we do not need to worry about stratification
    if isReg:
        # copy over the set of targets and find size
        subsetTable = dataSet.copy()
        subsetSize = round(len(subsetTable) * splitPercentage)

        # take a random sample of the subset with size as defined above, add to output
        set1Subset = subsetTable.sample(n=subsetSize)
        set1 = pd.concat([set1, set1Subset])

        # find what indices were not added to first set and add to second
        set2Indices = subsetTable.index.difference(set1.index)
        set2Subset = subsetTable.loc[set2Indices]
        set2 = pd.concat([set2, set2Subset])

    # otherwise, stratify by target output
    else:
        # find all unique outputs
        uniqueClassValues = (dataSet['Class'].unique().tolist())

        # iterate through each class
        for nextClass in uniqueClassValues:
            # subset by current class and find size to split off
            subsetTable = dataSet[dataSet['Class'] == nextClass]
            subsetSize = round(len(subsetTable) * splitPercentage)

            # sample from the class subset by size specified above and add to output
            set1Subset = subsetTable.sample(n=subsetSize)
            set1 = pd.concat([set1, set1Subset])

            # find the indices not added previously and add to the second data set
            set2Indices = subsetTable.index.difference(set1.index)
            set2Subset = subsetTable.loc[set2Indices]
            set2 = pd.concat([set2, set2Subset])

    return set1, set2


def testTree(currentTree, testSet):
    """
    This function will help a test record traverse a tree until it provides a prediction.

    @param currentTree: the tree to traverse for prediction
    @param testRecord: the record for which we will find the prediction for
    @return: the node that it reached
    """

    if currentTree.isReg is True:
        defaultPrediction = 0
    else:
        defaultPrediction = ''

    testResult = testSet.copy()
    testResult['leafDecisionNode'] = 0
    testResult['leafParentNode'] = 0
    testResult['classPrediction'] = defaultPrediction

    for rowIndex in testResult.index.tolist():

        currentNodeID = 0
        reachedLeaf = currentTree.treeTable.loc[currentNodeID, 'isLeaf']

        while not reachedLeaf:
            currentNodeID = findNextNode(currentTree, currentNodeID, testResult.loc[rowIndex])
            reachedLeaf = currentTree.treeTable.loc[currentNodeID, 'isLeaf']

        currentLeaf = currentNodeID
        testResult.loc[rowIndex, 'leafDecisionNode'] = currentLeaf
        testResult.loc[rowIndex, 'leafParentNode'] = currentTree.treeTable.loc[currentLeaf, 'parentNode']
        testResult.loc[rowIndex, 'classPrediction'] = currentTree.treeTable.loc[currentLeaf, 'nodePrediction']

    if currentTree.isReg is True:
        testResult['success'] = (testResult['classPrediction'] - testResult['Class'])**2
    else:
        testResult['success'] = testResult['classPrediction'] == testResult['Class']

    return testResult

def pruneTree(prunedTree, pruneSet):
    testResult = testTree(currentTree=prunedTree, testSet=pruneSet)
    prevSuccessRate = testResult['success'].mean()
    keepPruning = True
    loopCounter = 0

    while keepPruning:
        newPrunedTree = copy.deepcopy(prunedTree)

        parentNodeSuccessRate = testResult[['leafParentNode', 'success']].groupby(['leafParentNode'], as_index=False).mean('success')
        nodeToPrune = parentNodeSuccessRate.nsmallest(1, 'success').iat[0, 0]
        newPrunedTree.treeTable.loc[nodeToPrune, 'pruned'] = True
        newPrunedTree.treeTable.loc[nodeToPrune, 'isLeaf'] = True

        testResult = testTree(currentTree=newPrunedTree, testSet=pruneSet)
        currentSuccessRate = testResult['success'].mean()

        if prevSuccessRate > currentSuccessRate and loopCounter > 1:
            keepPruning = False
        else:
            prunedTree = newPrunedTree
            prevSuccessRate = currentSuccessRate
            loopCounter += 1

    return prunedTree

def findNextNode(currentTree, currentNodeID, testRecord):
    """
    This is a helper function for find prediction that helps us decide which node to progress to

    @param currentTree: the tree that we are traversing
    @param currentNodeID: the node that we are currently at
    @param testRecord: the record for which we are testing
    @return: the node ID for which we will move to
    """

    # pull out the pertinent information within the current node
    currentFeature = currentTree.treeTable.loc[currentNodeID, 'nodeFeature']
    currentChildren = eval(currentTree.treeTable.loc[currentNodeID, 'childrenNodes'])
    currentNodeFilters = eval(currentTree.treeTable.loc[currentNodeID, 'dataSetFilters'])

    # copy over our data and filter accordingly
    currentDataSubset = currentTree.trainData.copy()
    for currentFeature, featureType in currentNodeFilters.items():
        currentDataSubset = currentDataSubset[currentDataSubset[currentFeature] == featureType]

    # if the feature is a numeric value, need to create the relevant mean value to compare our value to
    if currentTree.featuresTypeMap[currentFeature] == 'Num':
        # find mean of train set
        trainMean = currentDataSubset[currentFeature].mean()

        # pick the branch based on its measure relative to the mean
        if trainMean > testRecord[currentFeature]:
            branchToTake = 'belowMean'
        else:
            branchToTake = 'aboveMean'

    # otherwise pick the categorical feature that it is equal to or closest to
    else:
        # find feature value of test record
        testFeatureAttribute = testRecord[currentFeature]

        # if the desired feature is available, pick that branch
        if testFeatureAttribute in currentChildren:
            branchToTake = testFeatureAttribute
        # otherwise, need to pick the closest value
        else:
            # for voting or sex, we pick a random value as "closest"
            if currentFeature == 'Sex' or currentTree.dataSetName == 'CongressVoting':
                branchToTake = np.random.choice(np.array(list(currentChildren.keys())), size=1)[0]
            # other categorical have some ordinal qualities, so we can find a closest
            else:
                branchToTake = findClosestCat(currentFeature, testFeatureAttribute, currentChildren)

    return currentChildren[branchToTake]

def findClosestCat(feature, featureValue, branchOptions):
    """
    This is a helper function for findNextNode to assist when a clear answer is not present. Tries to pick the closest value

    @param feature: the feature for which we are attempting to pick a path
    @param featureValue: the value for which we are trying to find the closest option
    @param branchOptions: the options available to us to move to
    @return: the option within available that is closest
    """

    # init our dictionary where we will store our differences
    differenceDictionary = {}

    # iterate through our branch options
    for key, value in branchOptions.items():
        # find absolute distance
        difference = abs(key - featureValue)

        # a few features need custom manipulation
        # if month, find true distance knowing it cant be more than 6 months
        if feature == 'month':
            if difference > 6:
                difference = 12 - difference
        # if day, find true distance knowing it cant be more than 3 days
        elif feature == 'day':
            if difference > 3:
                difference = 7 - difference

        # store the feature and its distance
        differenceDictionary[key] = difference

    # find the feature with the min distance
    closestCat = min(differenceDictionary, key=differenceDictionary.get)

    # return that feature
    return closestCat
