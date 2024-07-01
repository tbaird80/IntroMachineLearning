import pandas as pd
import numpy as np
from pandas import DataFrame

import AuxML2 as aux


class Tree:
    def __init__(self, dataName, isRegression, featuresMap, dataSet, existingTree=pd.DataFrame()):
        # init the attributes passed in to the function to be stored on tree object
        self.dataSetName = dataName
        self.featuresTypeMap = featuresMap
        self.isReg = isRegression
        self.trainData = dataSet.copy()

        # if not passing in existing tree data frame, then create new one
        if len(existingTree) == 0:
            self.createTree()
        # otherwise, grab the tree that was passed in
        else:
            self.treeTable = existingTree

    def createTree(self):
        self.treeTable = pd.DataFrame()

        # init our values to create tree
        keepAdding = True
        currentNode = 0

        # start by adding our root node
        self.addNode(newNodeID=currentNode, parentNodeID=-1)

        # continue running loop until there are no more nodes to add
        while keepAdding:
            # update the details of the current node
            self.evaluateNode(nodeID=currentNode)

            # check for
            remainingEvals: DataFrame = self.treeTable.copy()
            remainingEvals = remainingEvals[remainingEvals["evaluated"] == False]

            # if more nodes to add, find next node to add to
            if len(remainingEvals) > 0:
                currentNode = remainingEvals.head(1)["nodeID"].iloc[0]
            # otherwise, end our loop
            else:
                keepAdding = False

    def evaluateNode(self, nodeID):
        # copy our data and grab our relevant data filters
        currentDataSet = self.trainData.copy()
        currentNodeFilters = self.treeTable.loc[nodeID, 'dataSetFilters']

        # find the columns that we need to focus on, seeing diff between all possible and eval'd features
        allFeatures = list(self.featuresTypeMap.keys())
        usedFeatures = list(currentNodeFilters.keys())
        currentFeatures = np.setdiff1d(allFeatures, usedFeatures).tolist()

        # adjust our continuous variables relative to the mean of the train set at this point
        for featureToAdjust in allFeatures:

            # adjust if a numeric feature
            if self.featuresTypeMap[featureToAdjust] == 'Num':
                # find mean of train set
                currentMean = currentDataSet[featureToAdjust].mean()

                # update the values to either above or below mean to show binary split
                currentDataSet.loc[:, featureToAdjust] = np.where(currentDataSet[featureToAdjust] > currentMean, 'aboveMean', 'belowMean')

        # filter out the data as needed
        for currentFeature, featureType in currentNodeFilters.items():
            currentDataSet = currentDataSet[currentDataSet[currentFeature] == featureType]

        # see what possible outcomes are available
        uniqueClassesAvailable = currentDataSet['Class'].unique()

        # if either we only have 1 possible class or we are on our last filter, this is a leaf
        if len(uniqueClassesAvailable) == 1 or (len(currentNodeFilters) + 1) == len(self.featuresTypeMap):
            isNodeLeaf = True
        else:
            isNodeLeaf = False

        # init more descriptive values of our node
        childrenNodes = {}
        nextFeature = ""
        gainOrMSE = 0

        # if not leaf, need to add children nodes
        if not isNodeLeaf:
            # find next feature to pick, pull out the feature and its respective metric
            featureSummary = self.getNextFeature(currentDataSet, currentFeatures)
            nextFeature = featureSummary.loc[:, 'Variable'].iloc[0]
            gainOrMSE = featureSummary.loc[:, 'gainOrMSE'].iloc[0]

            # see what children options are available for the chosen feature
            nodeChildrenOptions = currentDataSet[nextFeature].unique()

            # iterate through the children, adding their init record and relevant features
            for currentChild in nodeChildrenOptions:
                # copy over the filters from current node to be added to child node
                childFilters = currentNodeFilters.copy()
                # update filter dictionary to account for the new filter
                childFilters[nextFeature] = currentChild

                # add the node to the table, add the new child node to the parent node record
                newChildNodeID = len(self.treeTable)
                self.addNode(newNodeID=newChildNodeID, parentNodeID=nodeID, dataFilters=childFilters)
                childrenNodes[currentChild] = newChildNodeID

        # find the prediction for the given node if leaf or eventually becomes leaf post pruning
        # regression values will take mean of remaining classes
        if self.isReg:
            nodePrediction = currentDataSet['Class'].mean()
        # categorical will take the most common class remaining
        else:
            nodePrediction = currentDataSet['Class'].value_counts().idxmax()

        # find the current tree level, if at root, level is 0, otherwise add 1 to parent level
        if nodeID == 0:
            currentTreeLevel = 0
        else:
            parentNodeID = self.treeTable.loc[nodeID, 'parentNode']
            currentTreeLevel = self.treeTable.loc[parentNodeID, 'treeLevel'] + 1

        # update the current node record with all the values that were calc'd above
        self.treeTable.loc[nodeID, 'nodeFeature'] = nextFeature
        self.treeTable.loc[nodeID, 'nodePrediction'] = nodePrediction
        self.treeTable.loc[nodeID, 'isLeaf'] = isNodeLeaf
        self.treeTable.at[nodeID, 'childrenNodes'] = childrenNodes
        self.treeTable.loc[nodeID, 'gainOrMSE'] = gainOrMSE
        self.treeTable.loc[nodeID, 'treeLevel'] = currentTreeLevel
        self.treeTable.loc[nodeID, 'evaluated'] = True

    def addNode(self, newNodeID, parentNodeID, dataFilters={}):
        if self.isReg:
            initNodePrediction = 0
        else:
            initNodePrediction = ""

        newRow = pd.DataFrame({
            'nodeID': [newNodeID],
            'parentNode': [parentNodeID],
            'dataSetFilters': [dataFilters],
            'evaluated': [False],
            'nodeFeature': [""],
            'nodePrediction': [initNodePrediction],
            'childrenNodes': [{}],
            'gainOrMSE': [0],
            'treeLevel': [0],
            'pruned': [False],
            'isLeaf': [False]
        }, index=[newNodeID])

        newRow['dataSetFilters'] = newRow['dataSetFilters'].astype('object')
        newRow['childrenNodes'] = newRow['childrenNodes'].astype('object')

        self.treeTable = pd.concat([self.treeTable, newRow])

    def getNextFeature(self, dataSet, currentFeatures):
        currentColumns = currentFeatures.copy()
        currentColumns.append('Class')

        # subset our data set by the features that we need
        dataSet = dataSet.copy()
        dataSet = dataSet[currentColumns]

        if self.isReg:
            # melt our dataset to show all the records in question
            meltedDataSet = pd.melt(dataSet, id_vars=['Class'], value_vars=currentFeatures, var_name='Variable', value_name='Value')

            # find the mean of each specific record
            regressionEstimateByClass = meltedDataSet.groupby(['Variable', 'Value'], as_index=False).mean('Class')
            regressionEstimateByClass.rename(columns={"Class": "estimatedOutput"}, inplace=True)

            # merge in the regression estimate for each of the possible values
            mseCalc = meltedDataSet.merge(regressionEstimateByClass, how='left', on=['Class', 'Variable', 'Value'])

            # calc our MSE
            mseCalc['gainOrMSE'] = (mseCalc['estimatedOutput'] - mseCalc['Class']) ** 2
            mseOutput = mseCalc.groupby(['Variable'], as_index=False).mean('gainOrMSE')

            # return the one with the lowest MSE
            return mseOutput.nsmallest(1, 'gainOrMSE')
        else:
            # fine unique class values
            uniqueClasses = dataSet['Class'].unique()

            # create a table to show all possible outcomes
            entropyOutput = pd.DataFrame({'Class': [], 'Variable': [], 'Value': []})

            for currentClass in uniqueClasses:
                for currentFeature in currentFeatures:
                    for currentFeatureValue in dataSet[currentFeature].unique():
                        newRow = pd.DataFrame({'Class': currentClass, 'Variable': currentFeature, 'Value': currentFeatureValue}, index=[len(entropyOutput)])
                        entropyOutput = pd.concat([entropyOutput, newRow], ignore_index=False)

            # melt our dataset to show all the records in question
            meltedDataSet = pd.melt(dataSet, id_vars=['Class'], value_vars=currentFeatures, var_name='Variable', value_name='Value')
            meltedDataSet['Count'] = 1

            # find the count of each specific record
            bottomLevelStats = meltedDataSet.groupby(['Class', 'Variable', 'Value'], as_index=False).sum('Count')
            bottomLevelStats.rename(columns={"Count": "bottomCount"}, inplace=True)

            # roll up the bottom level not accounting for class outcome
            middleLevelStats = meltedDataSet.groupby(['Variable', 'Value'], as_index=False).sum('Count')
            middleLevelStats.rename(columns={"Count": "middleCount"}, inplace=True)

            # find the count by class outcome
            topLevelStats = meltedDataSet.groupby(['Variable'], as_index=False).sum('Count')
            topLevelStats.rename(columns={"Count": "topCount"}, inplace=True)

            # find the top level entropy as starting point for gain ratio
            classLevelStats = meltedDataSet.groupby(['Class', 'Variable'], as_index=False).sum('Count')
            classLevelStats.rename(columns={"Count": "classCount"}, inplace=True)
            classLevelStats['topEntropy'] = -1 * classLevelStats['classCount'] / len(dataSet) * np.log2(classLevelStats['classCount'] / len(dataSet))
            classLevelStats = classLevelStats.groupby(['Variable'], as_index=False).sum('topEntropy')

            # merge in all the levels of data points
            entropyOutput = entropyOutput.merge(bottomLevelStats, how='left', on=['Class', 'Variable', 'Value'])
            entropyOutput = entropyOutput.merge(middleLevelStats, how='left', on=['Variable', 'Value'])
            entropyOutput = entropyOutput.merge(topLevelStats[['Variable', 'topCount']], how='left', on=['Variable'])

            # fill nas with very small number to not break log calc
            entropyOutput = entropyOutput.fillna(.000001)

            # calculate entropy at feature type level
            entropyOutput['bottomEntropy'] = -1 * (entropyOutput['bottomCount'] / entropyOutput['middleCount']) * np.log2(entropyOutput['bottomCount'] / entropyOutput['middleCount'])

            # calculate the entropy at the feature level
            featureSpecific = entropyOutput.groupby(['Variable', 'Value', 'middleCount', 'topCount'], as_index=False).sum('bottomEntropy')
            featureSpecific['middleEntropy'] = featureSpecific['middleCount'] / featureSpecific['topCount'] * featureSpecific['bottomEntropy']

            # calculate our IV function to correct the gain ratio for features with a lot of possibilities
            featureSpecific['IV'] = -1 * featureSpecific['middleCount'] / featureSpecific['topCount'] * np.log2(featureSpecific['middleCount'] / featureSpecific['topCount'])

            # roll up the middle entropy by the feature
            featureSpecific = featureSpecific[['Variable', 'middleEntropy', 'IV']].groupby(['Variable'], as_index=False).sum(['middleEntropy', 'IV'])

            # merge in the class level entropy
            featureSpecific = featureSpecific.merge(classLevelStats, how='left', on=['Variable'])

            # calc the numerator of ratio and ratio to pick the best next feature
            featureSpecific['gain'] = featureSpecific['topEntropy'] - featureSpecific['middleEntropy']
            featureSpecific['gainOrMSE'] = featureSpecific['gain'] / featureSpecific['IV']

            # return the largest gain ratio
            return featureSpecific.nlargest(1, 'gainOrMSE')





