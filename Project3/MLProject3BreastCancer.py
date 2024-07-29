import DataML3
from datetime import datetime
import os
import pandas as pd
import NeuralNetworkClass as network
import AuxML3 as aux

if __name__ == '__main__':
    """
    This is our main function for the Breast Cancer test set. It will train neural networks as simple regressions, multi-layer
    hidden node networks, and we will try out an autoencoder to see if we can condense out input layer.

    """
    pd.set_option('display.max_columns', None)
    # function inputs
    # data set title
    dataTitle = 'BreastCancer'

    # grab data
    features, targets = DataML3.dataSourcing(dataTitle)
    dataSet = features.join(targets)

    # define whether it is a regression
    regression = False

    # define the columns that need to be normalized
    normalCol = features.columns

    # tune data set
    # aux.tuneNetwork(dataSetName=dataTitle, fullDataSet=dataSet, isReg=regression, normalCol=normalCol, networkType="Simple")
    # aux.tuneNetwork(dataSetName=dataTitle, fullDataSet=dataSet, isReg=regression, normalCol=normalCol, networkType="BackPro")

    # run test
    testSimple = aux.runTest(dataSetName=dataTitle, fullDataSet=dataSet, isReg=regression, normalCol=normalCol, networkType="Simple", isTune=False)
    # testBackPro = aux.runTest(dataSetName=dataTitle, fullDataSet=dataSet, isReg=regression, normalCol=normalCol, networkType="BackPro", isTune=False)

    # train autoencoder
    aux.runWithAutoEncoder(dataSetName=dataTitle, fullDataSet=dataSet, isReg=regression, normalCol=normalCol, networkType="AutoEncoder")
