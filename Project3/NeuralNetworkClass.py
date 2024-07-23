import pandas as pd
import numpy as np
import math
import random
import copy

class NNet:
    def __init__(self, dataSetName, isRegression, trainingData, normalCols, numHiddenLayers, numHiddenLayerNodes, networkType):
        # set variables that define the network
        self.dataName = dataSetName
        self.isReg = isRegression
        self.normalCols = normalCols
        self.numHiddenLayers = numHiddenLayers
        self.numHiddenLayersNodes = numHiddenLayerNodes
        self.networkType = networkType

        # data to train on
        self.trainData = trainingData.copy()

        # init the traversal variables
        self.network = pd.DataFrame()
        self.autoencoder = pd.DataFrame()
        self.inputNames = trainingData.columns

        # create input layer
        inputNodeNames = list(self.trainData.columns)
        inputNodeNames.remove("Class")
        for currentInputName in inputNodeNames:
            inputLayerID = 0
            self.createNode(nodeName=currentInputName, nodeLayer=inputLayerID, weightMap={})
            self.createNodeAutoencoder(nodeName=currentInputName, nodeLayer=inputLayerID, weightMap={})

        # create hidden layers
        for currentHiddenLayer in range(0, numHiddenLayers):
            currentLayerID = currentHiddenLayer + 1

            newWeightMap = {'Intercept': random.uniform(-.01, .01)}
            prevNodeNames = self.network[self.network['nodeLayer'] == (currentLayerID - 1)]['nodeName'].tolist()
            for currentPrevName in prevNodeNames:
                newWeightMap[currentPrevName] = random.uniform(-.01, .01)

            for currentHiddenNode in range(0, numHiddenLayerNodes):
                currentNodeName = 'HiddenNode-' + str(currentHiddenNode) + "-" + str(currentHiddenLayer)
                self.createNode(nodeName=currentNodeName, nodeLayer=currentLayerID, weightMap=newWeightMap)

                if currentHiddenNode == 1:
                    self.createNodeAutoencoder(nodeName=currentNodeName, nodeLayer=currentLayerID, weightMap=newWeightMap)

        # create output layer
        outputLayerID = numHiddenLayers + 1

        # create mapping to previous layer with weights randomly created starting with intercept
        outputWeightMap = {'Intercept': random.uniform(-.01, .01)}
        prevNodeNames = self.network[self.network['nodeLayer'] == (outputLayerID - 1)]['nodeName'].tolist()
        for currentPrevName in prevNodeNames:
            outputWeightMap[currentPrevName] = random.uniform(-.01, .01)

        if self.isReg:
            possibleOutputs = ['OutputNode1']
        else:
            possibleOutputs = self.trainData['Class'].unique()

        for currentOutput in possibleOutputs:
            if self.isReg:
                currentNodeName = 'OutputNode'
            else:
                currentNodeName = currentOutput
            self.createNode(nodeName=currentNodeName, nodeLayer=outputLayerID, weightMap=outputWeightMap)

        # output layer is 2 in autoencoder, they match the inputs
        outputLayerID = 2
        inputNodeNames = list(self.trainData.columns)
        inputNodeNames.remove("Class")
        for currentInputName in inputNodeNames:
            # create mapping to previous layer with weights randomly created starting with intercept
            outputWeightMap = {'Intercept': random.uniform(-.01, .01)}
            prevNodeNames = self.network[self.network['nodeLayer'] == (outputLayerID - 1)]['nodeName'].tolist()

            # iterate through the previous layer names
            for currentPrevName in prevNodeNames:
                outputWeightMap[currentPrevName] = random.uniform(-.01, .01)

            self.createNodeAutoencoder(nodeName=currentInputName, nodeLayer=outputLayerID, weightMap=outputWeightMap)

    def createNode(self, nodeName, nodeLayer, weightMap):
        newRow = pd.DataFrame({
            'nodeName': [nodeName],
            'nodeLayer': [nodeLayer],
            'weightMap': [weightMap],
            'currentInputs': [[]],
            'currentOutputs': [[]],
            'actualValue': [[]],
            'currentPartialError': [[]]
        }, index=[len(self.network)])

        self.network = pd.concat([self.network, newRow])

    def createNodeAutoencoder(self, nodeName, nodeLayer, weightMap):
        newRow = pd.DataFrame({
            'nodeName': [nodeName],
            'nodeLayer': [nodeLayer],
            'weightMap': [weightMap],
            'currentInputs': [[]],
            'currentOutputs': [[]],
            'actualValue': [[]],
            'currentPartialError': [[]]
        }, index=[len(self.autoencoder)])

        self.autoencoder = pd.concat([self.autoencoder, newRow])

    def updateWithAutoencoder(self):
        # find the first two layers of autoencoder, and second two layers of regular network
        firstTwoLayers = self.autoencoder[self.autoencoder['nodeLayer'] < 2]
        lastTwoLayers = self.network[self.network['nodeLayer'] > 1]

        # concat the two sections together
        self.network = pd.concat([firstTwoLayers, lastTwoLayers], axis=0)

    def forwardPass(self, currentInputBatch):
        # iterate through all passed in records
        for currentInputID in currentInputBatch.index.tolist():
            # find next record
            currentInputRecord = currentInputBatch.loc[[currentInputID]]

            # grab all layers to iterate through, removing the first as an option
            layerIndices = self.network['nodeLayer'].unique().tolist()
            layerIndices.remove(0)

            # find the current class to compare against, then remove that column from the record
            currentClass = currentInputRecord['Class'].iloc[0]
            currentInputRecord = currentInputRecord.drop(['Class'], axis=1)

            # init dict to store classification softmax values
            softmaxOutputs = {}

            # update our input layers
            inputIndex = self.network[self.network['nodeLayer'] == 0].index.tolist()
            for currentIndex in inputIndex:
                currentName = self.network.loc[currentIndex, 'nodeName']
                self.network.loc[currentIndex, 'currentOutputs'].append(currentInputRecord[currentName].iloc[0])

            for currentLayer in layerIndices:
                # find the nodes relevant to that layer
                currentLayerNodes = self.network[self.network['nodeLayer'] == currentLayer]

                # grab previous layer outputs as the last item in their store list of outputs
                prevLayerOutputs = self.network[self.network['nodeLayer'] == currentLayer - 1][['nodeName', 'currentOutputs']].set_index('nodeName')
                prevLayerOutputs['currentOutputs'] = prevLayerOutputs['currentOutputs'].apply(lambda x: x[-1])

                # add bias input term
                prevLayerOutputs = prevLayerOutputs.append({'nodeName': 'Intercept', 'currentOutputs': 1})

                # iterate through all the nodes in that layer
                for currentNode in currentLayerNodes.index.tolist():
                    # grab the current weight map
                    weightMap = self.network.loc[currentNode, 'weightMap']

                    # convert the weight map to a dataframe
                    dictionaryTable = pd.DataFrame.from_dict(weightMap, orient='index')

                    # combine our weight map to our input values
                    mappingTable = pd.concat([prevLayerOutputs, dictionaryTable], axis=1)
                    mappingTable.columns = ['Inputs', 'Weights']

                    inputValue = (mappingTable['Inputs'] * mappingTable['Weights']).sum()

                    self.network.loc[currentNode, 'currentInputs'].append(inputValue)

                    # if output layer, add in our estimate and actual outputs
                    if currentLayer == self.numHiddenLayers + 1:
                        # if reg, add the estimated value and class value
                        if self.isReg:
                            self.network.loc[currentNode, 'currentOutputs'].append(math.exp(inputValue))
                            self.network.loc[currentNode, 'actualValue'].append(currentClass)
                        # otherwise add our target output and softmax numerator to be normalized after node processing complete
                        else:
                            softmaxOutputs[currentNode] = math.exp(inputValue)
                            self.network.loc[currentNode, 'actualValue'].append(int(currentClass == self.network.loc[currentNode, 'nodeName']))
                    else:
                        self.network.loc[currentNode, 'currentOutputs'].append(1 / (1 + math.exp(inputValue)))

                    # add our normalized softmax values as outputs
                if not self.isReg:
                    totalSoftmax = sum(softmaxOutputs.values())
                    for nodeID, softmaxValue in softmaxOutputs.items():
                        softmaxValue = softmaxValue / totalSoftmax
                        self.network.loc[nodeID, 'currentOutputs'].append(softmaxValue)

    def updatePartialErrors(self, indexStoppingPoint=0):
        # iterate through our network backwards
        currentLayer = self.numHiddenLayers + 1

        # check layer to stop updating at
        while currentLayer > indexStoppingPoint:
            # find our current layer of nodes
            currentLayerNodes = self.network[self.network['nodeLayer'] == currentLayer]
            nextLayerNodes = self.network[self.network['nodeLayer'] == currentLayer + 1]

            # iterate through our current layer nodes
            for currentNode in currentLayerNodes.index.tolist():
                # start with the last layer
                if currentLayer == self.numHiddenLayers + 1:
                    # grab our list of test cases to compare actual versus estimated to get mean difference
                    currentActualList = np.array(currentLayerNodes[currentNode, 'actualValue'])
                    currentOutputList = np.array(currentLayerNodes[currentNode, 'currentOutputs'])
                    outputError = currentActualList - currentOutputList
                    self.network.loc[currentNode, 'currentPartialError'] = outputError

                else:
                    currentNodeName = currentLayerNodes.loc[currentNode, 'nodeName']
                    nextLayerWeightedError = 0
                    for currentNextNode in nextLayerNodes.index.tolist():
                        currentNextWeightMap = nextLayerNodes.loc[currentNextNode, 'weightMap']
                        nextLayerWeightedError += currentNextWeightMap[currentNodeName] * nextLayerNodes.loc[currentNextNode, 'currentPartialError']

                    currentNodeOutputs = np.array(currentLayerNodes.loc[currentNode, 'currentOutputs'])
                    self.network.loc[currentNode, 'currentPartialError'] = nextLayerWeightedError * currentNodeOutputs * (1 - currentNodeOutputs)

            # move back one layer
            currentLayer -= 1

    def updateWeights(self, indexStoppingPoint=0):

        # create our learning rate as proportion of current set size (proxy size of output from first record) to entire training set
        currentTestSize = len(self.network['currentOutputs'].iloc(0))
        trainSetSize = len(self.trainData)
        currentLearningRate = currentTestSize / trainSetSize

        # iterate through our network backwards
        currentLayer = self.numHiddenLayers + 1
        while currentLayer > indexStoppingPoint:
            # iterate through our current layer nodes
            currentLayerNodes = self.network[self.network['nodeLayer'] == currentLayer]
            for currentNode in currentLayerNodes.index.tolist():
                # grab relevant weight map and partial error to help update our weights
                currentWeightMap = currentLayerNodes.loc[currentNode, 'weightMap']
                currentPartialError = np.array(currentLayerNodes.loc[currentNode, 'currentPartialError'])

                # iterate through the weights to update by learning rate and current partial error
                for inputName, inputWeight in currentWeightMap.items():
                    # find the previous layer outputs correcting manually for intercept to get our mean adjustment value
                    if inputName == 'Intercept':
                        averageAdjustment = np.mean(currentPartialError)
                    else:
                        previousLayerOutput = np.array(self.network[self.network['nodeName'] == inputName, 'currentOutputs'].iloc(0))
                        averageAdjustment = np.mean(currentPartialError * previousLayerOutput)
                    # adjust the weight by this calculated value
                    currentWeightMap[inputName] = inputWeight + currentLearningRate * averageAdjustment

            # move back one layer
            currentLayer -= 1

        # reset out output values once our weights have been updated
        self.dropLastForwardPass()

    def dropLastForwardPass(self):
        # reset out output values once our weights have been updated
        self.network['currentInputs'] = [[]]
        self.network['currentOutputs'] = [[]]
        self.network['actualValue'] = [[]]
        self.network['currentPartialError'] = [[]]
