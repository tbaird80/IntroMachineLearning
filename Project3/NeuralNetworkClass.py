import pandas as pd
import numpy as np


class NNet:
    def __init__(self, dataSetName, isRegression, trainingData, normalCols, numHiddenLayers, numHiddenLayerNodes):
        # set variables that define the network
        self.dataName = dataSetName
        self.isReg = isRegression
        self.normalCols = normalCols

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
            'currentInputs': [[]],
            'currentOutputs': [[]],
            'currentErrors': [[]]
        }, index=[len(self.network)])

        self.network = pd.concat([self.network, newRow])

    def testRecord(self, currentInputRecord):
        # grab all layers to iterate through, removing the first as an option
        layerIndices = self.network['nodeLayer'].unique().tolist()
        layerIndices.remove(0)

        # find the current class to compare against, then remove that column from the record
        currentClass = currentInputRecord['Class'].iloc[0]
        currentInputRecord = currentInputRecord.drop(['Class'], axis=1)

        # update our input layers
        inputIndex = self.network[self.network['nodeLayer'] == 0].index.tolist()
        for currentIndex in inputIndex:
            currentName = self.network.loc[currentIndex, 'nodeName']
            self.network.loc[currentIndex, 'currentOutputs'].append(currentInputRecord[currentName].iloc[0])

        for currentLayer in layerIndices:
            # find the nodes relevant to that layer
            currentLayerNodes = self.network[self.network['nodeLayer'] == currentLayer]

            # grab previous layer outputs
            prevLayerOutputs = self.network[self.network['nodeLayer'] == currentLayer - 1][['nodeName', 'currentOutputs']].set_index('nodeName')
            prevLayerOutputs['currentOutputs'] = prevLayerOutputs['currentOutputs'].apply(lambda x: x[-1])
            # iterate through all the nodes in that layer
            for currentNode in currentLayerNodes.index.tolist():

                # grab the current weight map
                weightMap = self.network.loc[currentNode, 'weightMap']

                # convert the weight map to a dataframe
                dictionaryTable = pd.DataFrame.from_dict(weightMap, orient='index')

                # combine our weight map to our input values
                mappingTable = pd.concat([prevLayerOutputs, dictionaryTable], axis=1)
                mappingTable.columns = ['Inputs', 'Weights']

                self.network.loc[currentNode, 'currentInputs'].append((mappingTable['Inputs'] * mappingTable['Weights']).sum())

                # find the output
                if currentLayer == layerIndices.max():
                    if self.isReg:
                        self.network.loc[currentNode, 'currentOutputs'].append((mappingTable['Inputs'] * mappingTable['Weights']).sum())
                        self.network.loc[currentNode, 'currentError'].append((self.network.loc[currentNode, 'currentValue'] - currentClass)**2)
                    # TODO: fix the logic for classification
                    else:
                        self.network.loc[currentNode, 'currentOutputs'].append((mappingTable['Inputs'] * mappingTable['Weights']).sum())
                        self.network.loc[currentNode, 'currentError'] = self.network.loc[currentNode, 'currentValue'] == currentClass
                else:
                    if self.isReg:
                        self.network.loc[currentNode, 'currentOutputs'].append((mappingTable['Inputs'] * mappingTable['Weights']).sum())
                    # TODO: fix the logic for classification
                    else:
                        self.network.loc[currentNode, 'currentOutputs'].append((mappingTable['Inputs'] * mappingTable['Weights']).sum())

