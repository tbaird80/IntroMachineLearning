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
    dataTitle = 'ComputerHardware'

    # grab data
    features, targets = DataML3.dataSourcing(dataTitle)
    dataSet = features.join(targets)

    # define whether it is a regression
    regression = True

    # define the columns that need to be normalized
    normalCol = ['MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX']

    # run test
    # testSimple = aux.runTest(dataSetName=dataTitle, fullDataSet=dataSet, isReg=regression, normalCol=normalCol, networkType="Simple")
    testBackPro = aux.runTest(dataSetName=dataTitle, fullDataSet=dataSet, isReg=regression, normalCol=normalCol, networkType="BackPro",  numHiddenLayerNodes=4)
