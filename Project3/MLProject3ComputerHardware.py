import DataML3
from datetime import datetime
import os
import pandas as pd
import NeuralNetworkClass as network


if __name__ == '__main__':
    """
    This is our main function for the Computer Hardware test set. It will define, prune, and test the set to return an effectiveness
    value for the decision tree algorithm. If you would like to run yourself, I would recommend doing so in chunks. Create full tree first,
    prune tree, test full tree, test prune tree, and then compare results.

    """
    # function inputs
    # data set title
    dataTitle = 'ComputerHardware'

    # grab data
    features, targets = DataML3.dataSourcing(dataTitle)
    dataSet = features.join(targets)

    # define whether it is a regression
    regression = True

    # train set
    trainSet = dataSet.iloc[[0]]

    # create our neural network
    basicRegression = network.NNet(dataSetName=dataTitle, isRegression=regression, trainingData=trainSet, numHiddenLayers=0, numHiddenLayerNodes=0)

    # test our neural network
    basicRegression.testNetwork(trainSet)

    print(basicRegression.network)

    #outputValue = basicRegression.testMe()
