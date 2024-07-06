import DataML2
from datetime import datetime
import os
import pandas as pd
import TreeClass
import AuxML2 as aux

if __name__ == '__main__':
    '''
    This is our main function for the Congress Voting test set. It will define, prune, and test the set to return an effectiveness
    value for the decision tree algorithm. If you would like to run yourself, I would recommend doing so in chunks. Create full tree first,
    prune tree, test full tree, test prune tree, and then compare results.

    '''
    # function inputs
    # data set title
    dataTitle = 'CongressVoting'

    # grab data
    features, targets = DataML2.dataSourcing(dataTitle)
    dataSet = features.join(targets)

    # define the features to be tuned
    featuresMap = {
        'handicapped-infants': 'Cat',
        'water-project-cost-sharing': 'Cat',
        'adoption-of-the-budget-resolution': 'Cat',
        'physician-fee-freeze': 'Cat',
        'el-salvador-aid': 'Cat',
        'religious-groups-in-schools': 'Cat',
        'anti-satellite-test-ban': 'Cat',
        'aid-to-nicaraguan-contras': 'Cat',
        'mx-missile': 'Cat',
        'immigration': 'Cat',
        'synfuels-corporation-cutback': 'Cat',
        'education-spending': 'Cat',
        'superfund-right-to-sue': 'Cat',
        'crime': 'Cat',
        'duty-free-exports': 'Cat',
        'export-administration-act-south-africa': 'Cat'
    }

    # define whether it is a regression
    regression = False

    # ----------------------Tests-----------------------------
    # create trees and return our test folder
    # testFolder = aux.runTreeCreation(dataTitle=dataTitle, dataSet=dataSet, featuresMap=featuresMap, isReg=regression)

    # test folder can either be passed from previous function or it can
    # currentTestDir = testFolder
    currentTestDir = dataTitle + "/UsedTestCases"

    # prune trees
    # aux.runTreePruning(dataTitle=dataTitle, featuresMap=featuresMap, isReg=regression, currentDir=currentTestDir)

    # test prePruned
    aux.runTreeTests(dataTitle=dataTitle, featuresMap=featuresMap, isReg=regression, currentDir=currentTestDir, isPrune=False)

    # test postPruned
    aux.runTreeTests(dataTitle=dataTitle, featuresMap=featuresMap, isReg=regression, currentDir=currentTestDir, isPrune=True)
