import pandas as pd
import numpy as np
import copy
import TreeClass
import os
from datetime import datetime

def runTreeCreation(dataTitle, dataSet, featuresMap, isReg):
    """
    A wrapper function that will create our original fully grown trees. It will create and store 10 trees along with the data sets that were used for training
    and the ones that will be used for pruning and testing.

    @param dataTitle: the overall data set for which we are working on
    @param dataSet: the dataset for which we will train our tree on
    @param featuresMap: the user defined mapping of the feature data types
    @param isReg: whether the tree is a regression tree or not
    @return: the directory for which our trees will be stored
    """

    # print start message of function
    print("***STARTING TREE CREATION: " + datetime.now().strftime("%d.%m.%Y_%I.%M.%S") + "***")

    # create directory based on current data set and timestamp
    currentTimestamp = datetime.now()
    timestampStr = currentTimestamp.strftime("%d.%m.%Y_%I.%M.%S")
    uniqueTestDir = dataTitle + "/" + timestampStr
    os.makedirs(uniqueTestDir)

    # init tree ID as we iterate through tree creation
    currentTreeID = 1

    # cycle through all 10 trees to be created, will increment 2 at a time
    while currentTreeID <= 10:
        # create 2 tree directories
        tree1DataDirectory = uniqueTestDir + "/Tree" + str(currentTreeID)
        tree2DataDirectory = uniqueTestDir + "/Tree" + str(currentTreeID + 1)
        os.makedirs(tree1DataDirectory)
        os.makedirs(tree2DataDirectory)

        # split our data set into 20% pruning data, 40% train1/test2, 40% train2/test1 (train or test depending on the index
        pruneSet, crossValidationSet = splitDataFrame(dataSet=dataSet, splitPercentage=.2, isReg=isReg)
        trainSet1, trainSet2 = splitDataFrame(dataSet=crossValidationSet, splitPercentage=.5, isReg=isReg)

        # create tree 1 of current iteration
        prePruneTree1 = TreeClass.Tree(dataName=dataTitle, isRegression=isReg, featuresMap=featuresMap, dataSet=trainSet1)

        # write our tree data and the relevant training/test/prune data to the directory
        treeFileName = tree1DataDirectory + "/prePruneTree.csv"
        prePruneTree1.treeTable.to_csv(treeFileName, index=True)
        trainDataFileName = tree1DataDirectory + "/trainData.csv"
        trainSet1.to_csv(trainDataFileName, index=True)
        testDataFileName = tree1DataDirectory + "/testData.csv"
        trainSet2.to_csv(testDataFileName, index=True)
        pruneDataFileName = tree1DataDirectory + "/pruneData.csv"
        pruneSet.to_csv(pruneDataFileName, index=True)

        # print status update
        print(dataTitle + " Tree " + str(currentTreeID) + " CREATED: " + datetime.now().strftime("%d.%m.%Y_%I.%M.%S"))

        # create tree 2 of current iteration
        prePruneTree2 = TreeClass.Tree(dataName=dataTitle, isRegression=isReg, featuresMap=featuresMap, dataSet=trainSet2)

        # write our tree data and the relevant training/test/prune data to the directory
        treeFileName = tree2DataDirectory + "/prePruneTree.csv"
        prePruneTree2.treeTable.to_csv(treeFileName, index=True)
        trainDataFileName = tree2DataDirectory + "/trainData.csv"
        trainSet2.to_csv(trainDataFileName, index=True)
        testDataFileName = tree2DataDirectory + "/testData.csv"
        trainSet1.to_csv(testDataFileName, index=True)
        pruneDataFileName = tree2DataDirectory + "/pruneData.csv"
        pruneSet.to_csv(pruneDataFileName, index=True)

        # print status update
        print(dataTitle + " Tree " + str(currentTreeID + 1) + " CREATED: " + datetime.now().strftime("%d.%m.%Y_%I.%M.%S"))

        # increment our tree ID by 2
        currentTreeID += 2

    # print success message and return directory for full STP where possible
    print("Trees created at: " + uniqueTestDir)
    return uniqueTestDir

def runTreePruning(dataTitle, featuresMap, isReg, currentDir):
    """
    Wrapper function that will take our fully grown trees and prune them to improve generality of our predictions.

    @param dataTitle: the data set for which we are testing
    @param featuresMap: the user defined mapping of the feature data types
    @param isReg: whether we are looking at a regression tree
    @param currentDir: the directory where the fully grown trees are stored
    """

    # print start message of function
    print("***STARTING PRUNING: " + datetime.now().strftime("%d.%m.%Y_%I.%M.%S") + "***")

    # iterate through all 10 trees created
    for currentTreeID in range(1, 11):
        # find next tree folder
        currentTreeFolder = currentDir + "/Tree" + str(currentTreeID)

        # grab relevant data to restore tree
        treeTable = pd.read_csv(currentTreeFolder + "/prePruneTree.csv", index_col=0)
        trainSet = pd.read_csv(currentTreeFolder + "/trainData.csv", index_col=0)
        pruneSet = pd.read_csv(currentTreeFolder + "/pruneData.csv", index_col=0)

        # restore and prune tree
        prunedTree = TreeClass.Tree(dataName=dataTitle, isRegression=isReg, featuresMap=featuresMap, dataSet=trainSet, existingTree=treeTable)
        newPrunedTree = pruneTree(prunedTree=prunedTree, pruneSet=pruneSet)

        # store pruned tree down
        prunedTreeFileName = currentTreeFolder + "/postPruneTree.csv"
        newPrunedTree.treeTable.to_csv(prunedTreeFileName, index=True)

        # print status update
        print(dataTitle + " Tree " + str(currentTreeID) + " PRUNED: " + datetime.now().strftime("%d.%m.%Y_%I.%M.%S"))

    # print success message
    print("All " + dataTitle + " trees pruned at: " + currentDir)

def runTreeTests(dataTitle, featuresMap, isReg, currentDir, isPrune):
    """
    Wrapper function to test both our fully grown and pruned trees.

    @param dataTitle: the data set for which we are testing
    @param featuresMap: the user defined mapping of the feature data types
    @param isReg: whether we are looking at a regression tree
    @param currentDir: the directory where our testable trees are stored
    @param isPrune: whether the test is on a pruned tree or a fully grown tree
    """

    # print start message of function
    if isPrune:
        print("***STARTING POST-PRUNE TESTS: " + datetime.now().strftime("%d.%m.%Y_%I.%M.%S") + "***")
    else:
        print("***STARTING PRE-PRUNE TESTS: " + datetime.now().strftime("%d.%m.%Y_%I.%M.%S") + "***")

    # init our test results output
    fullTestResults = pd.DataFrame()

    # find our test directory, create if it does not exist yet
    testResultDirectory = currentDir + "/TestResults"
    if not os.path.exists(testResultDirectory):
        os.makedirs(testResultDirectory)

    # iterate through all 10 trees
    for currentTreeID in range(1, 11):
        # grab current tree folder
        currentTreeFolder = currentDir + "/Tree" + str(currentTreeID)

        # grab train and test data
        trainSet = pd.read_csv(currentTreeFolder + "/trainData.csv", index_col=0)
        testSet = pd.read_csv(currentTreeFolder + "/testData.csv", index_col=0)

        # grab the tree needed, whether we are looking at prune tree or full tree
        if isPrune:
            treeTable = pd.read_csv(currentTreeFolder + "/postPruneTree.csv", index_col=0)
            currentTestResultsFile = testResultDirectory + "/postPrunedTreeResults.csv"
        else:
            treeTable = pd.read_csv(currentTreeFolder + "/prePruneTree.csv", index_col=0)
            currentTestResultsFile = testResultDirectory + "/prePrunedTreeResults.csv"

        # restore the tree and test it
        currentTree = TreeClass.Tree(dataName=dataTitle, isRegression=isReg, featuresMap=featuresMap, dataSet=trainSet, existingTree=treeTable)
        testResult = testTree(currentTree=currentTree, testSet=testSet)

        # add tree ID to results
        testResult['treeID'] = currentTreeID

        # add most recent test to output and print to file
        fullTestResults = pd.concat([fullTestResults, testResult])
        fullTestResults.to_csv(currentTestResultsFile, index=True)

        # print status update
        print(dataTitle + " Tree " + str(currentTreeID) + " TESTED: " + datetime.now().strftime("%d.%m.%Y_%I.%M.%S"))

    # update output message depending on the pruned nature of the tree
    if isPrune:
        outputMessage = "Full success rate of post-pruned trees: "
    else:
        outputMessage = "Full success rate of pre-pruned trees: "

    # output the success message with metrics
    print(outputMessage + str(fullTestResults['success'].mean()))
    print(fullTestResults[['treeID', 'success']].groupby(['treeID'], as_index=False).mean('success'))

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

def testTree(currentTree, testSet):
    """
    This function will help a test record traverse a tree until it provides a prediction.

    @param currentTree: the tree to traverse for prediction
    @return: the node that it reached
    """

    # set default prediction based on whether it will be a numeric outcome (reg) or character outcome (non-reg)
    if currentTree.isReg is True:
        defaultPrediction = 0
    else:
        defaultPrediction = ''

    # copy over our test set, init our output columns
    testResult = testSet.copy()
    testResult['leafDecisionNode'] = 0
    testResult['leafParentNode'] = 0
    testResult['classPrediction'] = defaultPrediction

    # iterate through all of our test cases
    for rowIndex in testResult.index.tolist():

        # start at the root
        currentNodeID = 0

        # init our check for if we reach a leaf to grab prediction from
        reachedLeaf = currentTree.treeTable.loc[currentNodeID, 'isLeaf']

        # iterate through the tree until we find a leaf
        while not reachedLeaf:
            currentNodeID = findNextNode(currentTree, currentNodeID, testResult.loc[rowIndex])
            reachedLeaf = currentTree.treeTable.loc[currentNodeID, 'isLeaf']

        # once we reach leaf, set the output values based on that leaf node
        currentLeaf = currentNodeID
        testResult.loc[rowIndex, 'leafDecisionNode'] = currentLeaf
        testResult.loc[rowIndex, 'leafParentNode'] = currentTree.treeTable.loc[currentLeaf, 'parentNode']
        testResult.loc[rowIndex, 'classPrediction'] = currentTree.treeTable.loc[currentLeaf, 'nodePrediction']

    # set success metric depending on whether it is reg or not
    if currentTree.isReg is True:
        testResult['success'] = (testResult['classPrediction'] - testResult['Class'])**2
    else:
        testResult['success'] = testResult['classPrediction'] == testResult['Class']

    return testResult

def pruneTree(prunedTree, pruneSet):
    """
    The function that will prune the passed in tree until performance begins to degrade.

    @param prunedTree: the tree to be pruned
    @param pruneSet: the data set that we are testing our result on
    @return: the pruned tree
    """

    # test tree and store down its performance as our starting point
    testResult = testTree(currentTree=prunedTree, testSet=pruneSet)
    prevSuccessRate = testResult['success'].mean()

    # init some loop control variables
    keepPruning = True
    loopCounter = 0

    # -----------------------------------------------------------------------------------------------------------------------------------------------
    # These are levers that we can pull depending on if we just want the pruning to have small forced amount (at least one) or a larger forced amount
    # (aim for 20% of prune-able nodes). These are somewhat arbitrary levers, but they do help in forcing the tree to prune to show a more generalized
    # model relative to the fully grown tree.
    # -----------------------------------------------------------------------------------------------------------------------------------------------
    # LEVER 1: at least 1
    numForcedPrune = 0

    # LEVER 2: aim for 20%
    # notLeaves = prunedTree.treeTable[['isLeaf']].value_counts()[False]
    # numForcedPrune = notLeaves * .2
    # -----------------------------------------------------------------------------------------------------------------------------------------------

    # continue pruning until otherwise set
    while keepPruning:
        # hard copy our current pruned tree to preserve previous pruning attempt
        newPrunedTree = copy.deepcopy(prunedTree)

        # see which node we should prune based on the set of children nodes that performed worst. We will prune at its parent
        parentNodeSuccessRate = testResult[['leafParentNode', 'success']].groupby(['leafParentNode'], as_index=False).mean('success')

        # if regression tree, worst performing is largest MSE
        if prunedTree.isReg:
            nodeToPrune = parentNodeSuccessRate.nlargest(1, 'success').iat[0, 0]
        # otherwise, worst performance is lowest success rate
        else:
            nodeToPrune = parentNodeSuccessRate.nsmallest(1, 'success').iat[0, 0]

        # prune the node, set as leaf
        newPrunedTree.treeTable.loc[nodeToPrune, 'pruned'] = True
        newPrunedTree.treeTable.loc[nodeToPrune, 'isLeaf'] = True

        # test the pruned tree and store its success rate
        testResult = testTree(currentTree=newPrunedTree, testSet=pruneSet)
        currentSuccessRate = testResult['success'].mean()

        # test for degradation as higher MSE in reg or lower success rate in non-reg
        if prunedTree.isReg:
            degradedPerformance = prevSuccessRate < currentSuccessRate
        else:
            degradedPerformance = prevSuccessRate > currentSuccessRate

        # set our stopping points as both performance degraded and we had performed our required number of pruned nodes
        if degradedPerformance and loopCounter > numForcedPrune:
            keepPruning = False
        # an extra stopping point would be if we hit the root as the node to prune
        elif nodeToPrune == 0:
            keepPruning = False
        # otherwise, point our current pruned tree to the new pruned tree, reset comparison success rate, and increment loop counter
        else:
            prunedTree = newPrunedTree
            prevSuccessRate = currentSuccessRate
            loopCounter += 1

    return prunedTree

def findNextNode(currentTree, currentNodeID, testRecord):
    """
    This is a helper function for find prediction that helps us decide which node to progress to

    @param currentTree: the tree that we are traversing
    @param currentNodeID: the node that we are currently at
    @param testRecord: the record for which we are testing
    @return: the node ID for which we will move to
    """

    # pull out the pertinent information within the current node
    currentFeature = currentTree.treeTable.loc[currentNodeID, 'nodeFeature']
    currentChildren = eval(currentTree.treeTable.loc[currentNodeID, 'childrenNodes'])
    currentNodeFilters = eval(currentTree.treeTable.loc[currentNodeID, 'dataSetFilters'])

    # copy over our data and filter accordingly
    currentDataSubset = currentTree.trainData.copy()
    for currentFeatureFilter, featureType in currentNodeFilters.items():

        # if the current feature filter is a numeric, then we need to adjust by the train data's mean
        if currentTree.featuresTypeMap[currentFeatureFilter] == 'Num':

            # grab train data mean
            currentFeatureTrainMean = currentDataSubset[currentFeatureFilter].mean()

            # check versus the mean depending on the given filter
            if featureType == 'aboveMean':
                currentDataSubset = currentDataSubset[currentDataSubset[currentFeatureFilter] > currentFeatureTrainMean]
            else:
                currentDataSubset = currentDataSubset[currentDataSubset[currentFeatureFilter] < currentFeatureTrainMean]

        # otherwise just match the value provided to filter
        else:
            currentDataSubset = currentDataSubset[currentDataSubset[currentFeatureFilter] == featureType]

    # if the feature is a numeric value, need to create the relevant mean value to compare our value to
    if currentTree.featuresTypeMap[currentFeature] == 'Num':
        # find mean of train set
        trainMean = currentDataSubset[currentFeature].mean()

        # pick the branch based on its measure relative to the mean, unless only one path in which case take the one path
        if len(currentChildren) == 1:
            branchToTake = list(currentChildren.keys())[0]
        elif trainMean > testRecord[currentFeature]:
            branchToTake = 'belowMean'
        else:
            branchToTake = 'aboveMean'

    # otherwise pick the categorical feature that it is equal to or closest to
    else:
        # find feature value of test record
        testFeatureAttribute = testRecord[currentFeature]

        # if the desired feature is available, pick that branch
        if testFeatureAttribute in currentChildren:
            branchToTake = testFeatureAttribute
        # otherwise, need to pick the closest value
        else:
            # for voting or sex, we pick a random value as "closest"
            if currentFeature == 'Sex' or currentTree.dataSetName == 'CongressVoting':
                branchToTake = np.random.choice(np.array(list(currentChildren.keys())), size=1)[0]
            # other categorical have some ordinal qualities, so we can find the closest
            else:
                branchToTake = findClosestCat(currentFeature, testFeatureAttribute, currentChildren)

    return currentChildren[branchToTake]

def findClosestCat(feature, featureValue, branchOptions):
    """
    This is a helper function for findNextNode to assist when a clear answer is not present. Tries to pick the closest value

    @param feature: the feature for which we are attempting to pick a path
    @param featureValue: the value for which we are trying to find the closest option
    @param branchOptions: the options available to us to move to
    @return: the option within available that is closest
    """

    # init our dictionary where we will store our differences
    differenceDictionary = {}

    # iterate through our branch options
    for key, value in branchOptions.items():
        # find absolute distance
        difference = abs(key - featureValue)

        # a few features need custom manipulation
        # if month, find true distance knowing it cant be more than 6 months
        if feature == 'month':
            if difference > 6:
                difference = 12 - difference
        # if day, find true distance knowing it cant be more than 3 days
        elif feature == 'day':
            if difference > 3:
                difference = 7 - difference

        # store the feature and its distance
        differenceDictionary[key] = difference

    # find the feature with the min distance
    closestCat = min(differenceDictionary, key=differenceDictionary.get)

    # return that feature
    return closestCat
