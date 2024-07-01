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

    # ----------------------Create Tree-----------------------------
    # Get the current timestamp and create our own unique new directory
    # currentTimestamp = datetime.now()
    # timestampStr = currentTimestamp.strftime("%d.%m.%Y_%I.%M.%S")
    # uniqueTestID = dataTitle + "/" + timestampStr
    # os.makedirs(uniqueTestID)
    #
    # pruneSet, crossValidationSet = aux.splitDataFrame(dataSet=dataSet, splitPercentage=.2, isReg=regression)
    # trainSet, testSet = aux.splitDataFrame(dataSet=crossValidationSet, splitPercentage=.5, isReg=regression)
    #
    # prePruneTree = TreeClass.Tree(dataName=dataTitle, isRegression=regression, featuresMap=featuresMap, dataSet=trainSet)
    #
    # currentTreeID = 1
    # currentTreeData = uniqueTestID + "/Tree" + str(currentTreeID)
    # os.makedirs(currentTreeData)
    #
    # treeFileName = currentTreeData + "/prePruneTree.csv"
    # trainDataFileName = currentTreeData + "/trainData.csv"
    # testDataFileName = currentTreeData + "/testData.csv"
    # pruneDataFileName = currentTreeData + "/pruneData.csv"
    #
    # prePruneTree.treeTable.to_csv(treeFileName, index=True)
    # trainSet.to_csv(trainDataFileName, index=True)
    # testSet.to_csv(testDataFileName, index=True)
    # pruneSet.to_csv(pruneDataFileName, index=True)

    # ------------------------Reset current test Folder----------------
    currentTestFolder = dataTitle + "/01.07.2024_12.07.34"

    # -----------------------Prune Tree------------------------------
    currentTreeFolder = currentTestFolder + "/Tree" + str(1)

    treeTable = pd.read_csv(currentTreeFolder + "/prePruneTree.csv")
    trainSet = pd.read_csv(currentTreeFolder + "/trainData.csv")
    pruneSet = pd.read_csv(currentTreeFolder + "/pruneData.csv")
    prunedTreeFileName = currentTestFolder + "/prePruneTree.csv"

    prunedTree = TreeClass.Tree(dataName=dataTitle, isRegression=regression, featuresMap=featuresMap, dataSet=trainSet, existingTree=treeTable)

    newPrunedTree = aux.pruneTree(prunedTree=prunedTree, pruneSet=pruneSet)

    newPrunedTree.treeTable.to_csv(prunedTreeFileName, index=True)

    # -----------------------Test Full Tree--------------------------

    prePrunedTestResults = pd.DataFrame()

    currentTreeFolder = currentTestFolder + "/Tree" + str(1)
    currentTestResultsFile = currentTestFolder + "/TestResults/prePrunedTreeResults.csv"

    treeTable = pd.read_csv(currentTreeFolder + "/prePruneTree.csv", index_col=0)
    trainSet = pd.read_csv(currentTreeFolder + "/trainData.csv", index_col=0)
    testSet = pd.read_csv(currentTreeFolder + "/testData.csv", index_col=0)

    prePruneTree = TreeClass.Tree(dataName=dataTitle, isRegression=regression, featuresMap=featuresMap, dataSet=trainSet, existingTree=treeTable)

    testResult = aux.testTree(currentTree=prePruneTree, testSet=testSet)

    testResult.to_csv(currentTestResultsFile, index=True)

    prePrunedTestResults = pd.concat([prePrunedTestResults, testResult])


    # -----------------------Test Pruned Tree------------------------