import pandas as pd
import numpy as np


class NNet:
    def __init__(self, dataSetName, isRegression, trainingData, numHiddenLayers, numHiddenLayerNodes):
        # set variables that define the network
        self.dataName = dataSetName
        self.isReg = isRegression

        # data to train on
        self.trainData = trainingData.copy()

        # init the traversal variables
        self.network = pd.DataFrame()
        self.inputNames = trainingData.columns

        # create input layer
        inputNodeNames = list(self.trainData.columns)
        inputNodeNames.remove("Class")
        for currentInputName in inputNodeNames:
            inputLayerID = 0
            self.createNode(nodeName=currentInputName, nodeLayer=inputLayerID, weightMap = {})

        # create hidden layers
        for currentHiddenLayer in range(0, numHiddenLayers):
            currentLayerID = currentHiddenLayer + 1
            newWeightMap = {}

            prevNodeNames = self.network[self.network['nodeLayer'] == (currentLayerID - 1)]['nodeName'].tolist()
            for currentPrevName in prevNodeNames:
                newWeightMap[currentPrevName] = .1

            for currentHiddenNode in range(0, numHiddenLayerNodes):
                currentNodeName = 'HiddenNode-' + str(currentHiddenNode) + "-" + str(currentHiddenLayer)
                self.createNode(nodeName=currentNodeName, nodeLayer=currentLayerID, weightMap=newWeightMap)

        # create output layer
        outputLayerID = numHiddenLayers + 1
        newWeightMap = {}

        prevNodeNames = self.network[self.network['nodeLayer'] == (outputLayerID - 1)]['nodeName'].tolist()
        for currentPrevName in prevNodeNames:
            newWeightMap[currentPrevName] = .1

        if self.isReg:
            self.createNode(nodeName='OutputNode1', nodeLayer=outputLayerID, weightMap=newWeightMap)
        else:
            possibleOutputs = self.trainData['Class'].unique()
            for currentOutput in possibleOutputs:
                currentNodeName = 'OutputNode-' + currentOutput
                self.createNode(nodeName=currentNodeName, nodeLayer=outputLayerID, weightMap=newWeightMap)

    def createNode(self, nodeName, nodeLayer, weightMap):
        newRow = pd.DataFrame({
            'nodeName': [nodeName],
            'nodeLayer': [nodeLayer],
            'weightMap': [weightMap],
            'currentValue': [0],
            'currentError': [0]
        }, index=[len(self.network)])

        self.network = pd.concat([self.network, newRow])

    def testNetwork(self, currentInputRecord):

        layerIndices = self.network['nodeLayer'].unique().tolist()
        layerIndices.remove(0)

        currentClass = currentInputRecord['Class'].iloc[0]
        currentInputRecord = currentInputRecord.drop(['Class'], axis=1)

        for currentLayer in layerIndices:
            currentLayerNodes = self.network[self.network['nodeLayer'] == currentLayer]

            for currentNode in currentLayerNodes.index.tolist():
                weightMap = self.network.loc[currentNode, 'weightMap']

                dictionaryTable = pd.DataFrame.from_dict(weightMap, orient='index')

                mappingTable = pd.concat([currentInputRecord.iloc[0], dictionaryTable], axis=1)
                mappingTable.columns = ['Inputs', 'Weights']

                self.network.loc[currentNode, 'currentValue'] = (mappingTable['Inputs'] * mappingTable['Weights']).sum()

                if self.isReg:
                    self.network.loc[currentNode, 'currentError'] = (self.network.loc[currentNode, 'currentValue'] - currentClass)**2
                else:
                    self.network.loc[currentNode, 'currentError'] = self.network.loc[currentNode, 'currentValue'] == currentClass
