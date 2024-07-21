import DataML3
from datetime import datetime
import os
import pandas as pd
import NeuralNetworkClass as network
import AuxML3 as aux

if __name__ == '__main__':
    """
    This is our main function for the Computer Hardware test set. It will define, prune, and test the set to return an effectiveness
    value for the decision tree algorithm. If you would like to run yourself, I would recommend doing so in chunks. Create full tree first,
    prune tree, test full tree, test prune tree, and then compare results.

    """
    pd.set_option('display.max_columns', None)
    # function inputs
    # data set title
    dataTitle = 'CongressVoting'

    # grab data
    features, targets = DataML3.dataSourcing(dataTitle)
    dataSet = features.join(targets)

    # define whether it is a regression
    regression = False

    # define the columns that need to be normalized
    normalCol = []

    # split into tune, test, and train sets
    tuneSet, trainTestSet = aux.splitDataFrame(dataSet, .2, regression)
    trainSet, testSet = aux.splitDataFrame(trainTestSet, .5, regression)

    normalizeDataSet = trainSet.copy()
    aux.normalizeNumberValues(tuneSet, normalizeDataSet, normalCol)
    aux.normalizeNumberValues(trainSet, normalizeDataSet, normalCol)
    aux.normalizeNumberValues(testSet, normalizeDataSet, normalCol)

    # create our neural network
    simpleNetwork = network.NNet(dataSetName=dataTitle, isRegression=regression, trainingData=trainSet,
                                 normalCols=normalCol, numHiddenLayers=0, numHiddenLayerNodes=0, networkType="Simple")

    print(trainSet.iloc[[0]])

    simpleNetwork.forwardPass(trainSet.iloc[[0]])

    print(simpleNetwork.network)

    # create our neural network
    backProNetwork = network.NNet(dataSetName=dataTitle, isRegression=regression, trainingData=trainSet,
                                  normalCols=normalCol, numHiddenLayers=2, numHiddenLayerNodes=3, networkType="BackPro")

    # test our neural network
    backProNetwork.forwardPass(trainSet.iloc[[0]])

    print(backProNetwork.network)
