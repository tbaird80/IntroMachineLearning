import DataML2
from datetime import datetime
import os
import pandas as pd
import AuxML2 as aux
import TreeClass

if __name__ == '__main__':
    '''
    This is our main function for the Car Evaluation test set. It will define, prune, and test the set to return an effectiveness
    value for the decision tree algorithm. If you would like to run yourself, I would recommend doing so in chunks. Create full tree first,
    prune tree, test full tree, test prune tree, and then compare results.

    '''
    # function inputs
    # data set title
    dataTitle = 'CarEval'

    # grab data
    features, targets = DataML2.dataSourcing(dataTitle)
    dataSet = features.join(targets)

    # define the features to be tuned
    featuresMap = {
        'buying': 'Cat',
        'maint': 'Cat',
        'doors': 'Cat',
        'persons': 'Cat',
        'lug_boot': 'Cat',
        'safety': 'Cat'
    }

    # define whether it is a regression
    regression = False

    # ----------------------Create Tree-----------------------------
    # Get the current timestamp and create our own unique new directory
    currentTimestamp = datetime.now()
    timestampStr = currentTimestamp.strftime("%d.%m.%Y_%I.%M.%S")
    uniqueTestID = dataTitle + "/" + timestampStr
    os.makedirs(uniqueTestID)

    pruneSet, crossValidationSet = aux.splitDataFrame(dataSet=dataSet, splitPercentage=.2, isReg=regression)
    trainSet, testSet = aux.splitDataFrame(dataSet=crossValidationSet, splitPercentage=.5, isReg=regression)

    unprundedTree = TreeClass.Tree(dataName=dataTitle, isRegression=regression, featuresMap=featuresMap, dataSet=trainSet)



    # -----------------------Prune Tree------------------------------

    # -----------------------Test Full Tree--------------------------

    # -----------------------Test Pruned Tree------------------------
