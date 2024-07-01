import DataML2
from datetime import datetime
import os
import pandas as pd
import TreeClass
import AuxML2 as aux

if __name__ == '__main__':
    '''
    This is our main function for the Forest Fire test set. It will define, prune, and test the set to return an effectiveness
    value for the decision tree algorithm. If you would like to run yourself, I would recommend doing so in chunks. Create full tree first,
    prune tree, test full tree, test prune tree, and then compare results.

    '''
    # function inputs
    # data set title
    dataTitle = 'ForestFires'

    # grab data
    features, targets = DataML2.dataSourcing(dataTitle)
    dataSet = features.join(targets)

    # define the features to be tuned
    featuresMap = {
        'X': 'Num',
        'Y': 'Num',
        'month': 'Cat',
        'day': 'Cat',
        'FFMC': 'Num',
        'DMC': 'Num',
        'DC': 'Num',
        'ISI': 'Num',
        'temp': 'Num',
        'RH': 'Num',
        'wind': 'Num',
        'rain': 'Num'
    }

    # define whether it is a regression
    regression = True

    # ----------------------Create Tree-----------------------------
    # Get the current timestamp and create our own unique new directory
    currentTimestamp = datetime.now()
    timestampStr = currentTimestamp.strftime("%d.%m.%Y_%I.%M.%S")
    uniqueTestID = dataTitle + "/" + timestampStr
    os.makedirs(uniqueTestID)

    for currentTreeID in range(1, 11):
        currentTreeData = uniqueTestID + "/Tree" + str(currentTreeID)
        os.makedirs(currentTreeData)

        pruneSet, crossValidationSet = aux.splitDataFrame(dataSet=dataSet, splitPercentage=.2, isReg=regression)
        trainSet, testSet = aux.splitDataFrame(dataSet=crossValidationSet, splitPercentage=.5, isReg=regression)

        prePruneTree = TreeClass.Tree(dataName=dataTitle, isRegression=regression, featuresMap=featuresMap, dataSet=trainSet)

        treeFileName = currentTreeData + "/prePruneTree.csv"
        trainDataFileName = currentTreeData + "/trainData.csv"
        testDataFileName = currentTreeData + "/testData.csv"
        pruneDataFileName = currentTreeData + "/pruneData.csv"

        prePruneTree.treeTable.to_csv(treeFileName, index=True)
        trainSet.to_csv(trainDataFileName, index=True)
        testSet.to_csv(testDataFileName, index=True)
        pruneSet.to_csv(pruneDataFileName, index=True)

    # ------------------------Reset current test Folder----------------
    currentTestFolder = dataTitle + "/01.07.2024_12.07.34"

    # -----------------------Prune Tree------------------------------
    for currentTreeID in range(1, 11):
        currentTreeFolder = currentTestFolder + "/Tree" + str(currentTreeID)

        treeTable = pd.read_csv(currentTreeFolder + "/prePruneTree.csv")
        trainSet = pd.read_csv(currentTreeFolder + "/trainData.csv")
        pruneSet = pd.read_csv(currentTreeFolder + "/pruneData.csv")
        prunedTreeFileName = currentTreeFolder + "/pruneTree.csv"

        prunedTree = TreeClass.Tree(dataName=dataTitle, isRegression=regression, featuresMap=featuresMap, dataSet=trainSet, existingTree=treeTable)

        newPrunedTree = aux.pruneTree(prunedTree=prunedTree, pruneSet=pruneSet)

        newPrunedTree.treeTable.to_csv(prunedTreeFileName, index=True)

    # -----------------------Test Full Tree--------------------------
    prePrunedTestResults = pd.DataFrame()

    for currentTreeID in range(1, 11):
        currentTreeFolder = currentTestFolder + "/Tree" + str(currentTreeID)
        currentTestResultsFile = currentTestFolder + "/TestResults/prePrunedTreeResults.csv"

        treeTable = pd.read_csv(currentTreeFolder + "/prePruneTree.csv", index_col=0)
        trainSet = pd.read_csv(currentTreeFolder + "/trainData.csv", index_col=0)
        testSet = pd.read_csv(currentTreeFolder + "/testData.csv", index_col=0)

        prePruneTree = TreeClass.Tree(dataName=dataTitle, isRegression=regression, featuresMap=featuresMap, dataSet=trainSet, existingTree=treeTable)

        testResult = aux.testTree(currentTree=prePruneTree, testSet=testSet)

        testResult['treeID'] = currentTreeID

        prePrunedTestResults = pd.concat([prePrunedTestResults, testResult])

        prePrunedTestResults.to_csv(currentTestResultsFile, index=True)

        print(testResult['success'].mean())

    # -----------------------Test Pruned Tree------------------------
    prunedTestResults = pd.DataFrame()

    for currentTreeID in range(1, 11):
        currentTreeFolder = currentTestFolder + "/Tree" + str(currentTreeID)
        currentTestResultsFile = currentTestFolder + "/TestResults/prunedTreeResults.csv"

        treeTable = pd.read_csv(currentTreeFolder + "/pruneTree.csv", index_col=0)
        trainSet = pd.read_csv(currentTreeFolder + "/trainData.csv", index_col=0)
        testSet = pd.read_csv(currentTreeFolder + "/testData.csv", index_col=0)

        pruneTree = TreeClass.Tree(dataName=dataTitle, isRegression=regression, featuresMap=featuresMap, dataSet=trainSet, existingTree=treeTable)

        testResult = aux.testTree(currentTree=pruneTree, testSet=testSet)

        testResult['treeID'] = currentTreeID

        prunedTestResults = pd.concat([prunedTestResults, testResult])

        prePrunedTestResults.to_csv(currentTestResultsFile, index=True)

        print(testResult['success'].mean())
