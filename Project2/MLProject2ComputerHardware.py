import DataML2
from datetime import datetime
import os
import pandas as pd
import TreeClass
import AuxML2 as aux

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
    features, targets = DataML2.dataSourcing(dataTitle)
    dataSet = features.join(targets)

    # define the features to be tuned
    featuresMap = {
        'MYCT': 'Num',
        'MMIN': 'Num',
        'MMAX': 'Num',
        'CACH': 'Num',
        'CHMIN': 'Num',
        'CHMAX': 'Num'
    }

    # define whether it is a regression
    regression = True

    # ----------------------Tests-----------------------------
    # create trees and return our test folder
    # testFolder = aux.runTreeCreation(dataTitle=dataTitle, dataSet=dataSet, featuresMap=featuresMap, isReg=regression)

    # test folder can either be passed from previous function or it can
    # currentTestDir = testFolder
    currentTestDir = dataTitle + "/07.07.2024_01.12.02"

    # prune trees
    # aux.runTreePruning(dataTitle=dataTitle, featuresMap=featuresMap, isReg=regression, currentDir=currentTestDir)

    # test prePruned
    aux.runTreeTests(dataTitle=dataTitle, featuresMap=featuresMap, isReg=regression, currentDir=currentTestDir, isPrune=False)

    # test postPruned
    # aux.runTreeTests(dataTitle=dataTitle, featuresMap=featuresMap, isReg=regression, currentDir=currentTestDir, isPrune=True)
