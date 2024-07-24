from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
import os
from datetime import datetime
import NeuralNetworkClass as network
import copy


def runTest(dataSetName, fullDataSet, isReg, normalCol, networkType, numHiddenLayerNodes=0):

    # create directory based on current data set and timestamp
    currentTimestamp = datetime.now()
    timestampStr = currentTimestamp.strftime("%d.%m.%Y_%I.%M.%S")
    uniqueTestDir = dataSetName + "/" + timestampStr
    # os.makedirs(uniqueTestDir)
    #
    # # current simple test directory
    # uniqueTestDirSimple = uniqueTestDir + "/" + networkType
    # os.makedirs(uniqueTestDirSimple)

    # update number of hidden nodes based on network type
    if networkType == 'Simple':
        numHiddenLayers = 0
        numHiddenLayerNodes = numHiddenLayerNodes
    else:
        numHiddenLayers = 2
        numHiddenLayerNodes = numHiddenLayerNodes

    # current network name
    currentNetworkID = 1

    # we want to run a 5x2, so create 5 loops
    while currentNetworkID <= 10:
        currentTune, currentTrain, currentTest = createTuneTrainTest(fullDataSet, isReg, normalCol)

        currentNetwork = network.NNet(dataSetName=dataSetName, isRegression=isReg, trainingData=currentTrain,
                                      normalCols=normalCol, numHiddenLayers=numHiddenLayers, numHiddenLayerNodes=numHiddenLayerNodes, networkType=networkType)

        currentNetwork.trainNetwork(currentTune, indexStop=0)

        currentNetworkID += 10

    return uniqueTestDir


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
            dataMin = trainSet[currentCol].min()
            dataMax = trainSet[currentCol].max()
            testSet[currentCol] = (testSet[currentCol] - dataMin) / (dataMax - dataMin)
        else:
            continue
    #return testSet

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

def createTuneTrainTest(dataSet, isReg, normalCols):
    # split into tune, test, and train sets
    tuneSet, trainTestSet = splitDataFrame(dataSet, .2, isReg)
    trainSet, testSet = splitDataFrame(trainTestSet, .5, isReg)

    normalizeDataSet = trainSet.copy()
    normalizeNumberValues(tuneSet, normalizeDataSet, normalCols)
    normalizeNumberValues(trainSet, normalizeDataSet, normalCols)
    normalizeNumberValues(testSet, normalizeDataSet, normalCols)

    return tuneSet, trainSet, testSet
