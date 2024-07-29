import pandas as pd
import numpy as np
import math
import random
import copy
from datetime import datetime
import AuxML3 as aux

class NNet:
    '''
    This class will help us create, test, and train neural networks to predict outcomes of provided data sets

    '''
    def __init__(self, dataSetName, isRegression, trainingData, normalCols, proportionHiddenNodesToInput, networkType):
        '''
        The constructor class of our neural network

        @param dataSetName: the data set name
        @param isRegression: whether it is a regression or not
        @param trainingData: the data which we will train our model on
        @param normalCols: the columns that we want to normalize
        @param proportionHiddenNodesToInput: the proportion of hidden nodes relative to the inputs
        @param networkType: the network that we want to build
        '''

        # calculate the number of hidden layers
        if networkType == 'Simple':
            numHiddenLayers = 0
        else:
            numHiddenLayers = 2

        # calculate the number of hidden nodes per layer
        numHiddenLayerNodes = int(len(trainingData.columns) * proportionHiddenNodesToInput)

        # set variables that define the network
        self.dataName = dataSetName
        self.isReg = isRegression
        self.normalCols = normalCols
        self.numHiddenLayers = numHiddenLayers
        self.numHiddenLayerNodes = numHiddenLayerNodes
        self.networkType = networkType

        # data to train on
        self.trainData = trainingData.copy()

        # init the traversal variables
        self.network = pd.DataFrame()
        self.autoencoder = pd.DataFrame()
        self.inputNames = trainingData.columns

        # train variables
        self.trainNetworkList = []
        self.trainPerformanceList = []

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

            for currentHiddenNode in range(0, numHiddenLayerNodes):
                # create new weight mapping based on previous outputs
                newWeightMap = {'Intercept': random.uniform(-.01, .01)}
                prevNodeNames = self.network[self.network['nodeLayer'] == (currentLayerID - 1)]['nodeName'].tolist()

                # iterate through the previous layer inputs and set random weight
                for currentPrevName in prevNodeNames:
                    newWeightMap[currentPrevName] = random.uniform(-.01, .01)

                # name our hidden node values and create them in the network
                currentNodeName = 'HiddenNode-' + str(currentHiddenNode) + "-" + str(currentHiddenLayer)
                self.createNode(nodeName=currentNodeName, nodeLayer=currentLayerID, weightMap=newWeightMap)

                # create our initial layer for our autoencoder
                if currentHiddenLayer == 0:
                    self.createNodeAutoencoder(nodeName=currentNodeName, nodeLayer=currentLayerID, weightMap=newWeightMap)

        # create output layer
        outputLayerID = numHiddenLayers + 1

        # create our possible outputs. just 1 node for reg, but multiple for classification
        if self.isReg:
            possibleOutputs = ['OutputNode1']
        else:
            possibleOutputs = self.trainData['Class'].unique()

        # iterate through our possible outpouts
        for currentOutput in possibleOutputs:
            # create mapping to previous layer with weights randomly created starting with intercept
            outputWeightMap = {'Intercept': random.uniform(-.01, .01)}
            prevNodeNames = self.network[self.network['nodeLayer'] == (outputLayerID - 1)]['nodeName'].tolist()

            # iterate through our prev layer values with randomly created weights
            for currentPrevName in prevNodeNames:
                outputWeightMap[currentPrevName] = random.uniform(-.01, .01)

            # name our output node depending on if reg or classification
            if self.isReg:
                currentNodeName = 'OutputNode'
            else:
                currentNodeName = currentOutput

            # create the next output node
            self.createNode(nodeName=currentNodeName, nodeLayer=outputLayerID, weightMap=outputWeightMap)

        # output layer is 2 in autoencoder, they match the inputs
        outputLayerID = 2
        inputNodeNames = list(self.trainData.columns)
        inputNodeNames.remove("Class")

        # for our autoencoder, we want to create outputs that match our inputs
        for currentInputName in inputNodeNames:

            # create mapping to first hidden layer with weights randomly created starting with intercept
            outputWeightMap = {'Intercept': random.uniform(-.01, .01)}
            prevNodeNames = self.network[self.network['nodeLayer'] == 1]['nodeName'].tolist()

            # create the random initial weights
            for currentPrevName in prevNodeNames:
                outputWeightMap[currentPrevName] = random.uniform(-.01, .01)

            self.createNodeAutoencoder(nodeName=currentInputName, nodeLayer=outputLayerID, weightMap=outputWeightMap)

    def createNode(self, nodeName, nodeLayer, weightMap):
        '''
        Create our next node for our network

        @param nodeName: the node that we want to name it as
        @param nodeLayer: the layer which the node is in
        @param weightMap: the randomly created weight map
        @return: nothing
        '''

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
        '''
        Create our next node for our autoencoder

        @param nodeName: the node that we want to name it as
        @param nodeLayer: the layer which the node is in
        @param weightMap: the randomly created weight map
        @return: nothing
        '''

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
        '''
        update our network to have the trained autoencoder as our first two layers

        '''
        # copy both networks
        currentAutoencoder = aux.hardCopyDataframe(self.autoencoder)
        currentNetwork = aux.hardCopyDataframe(self.network)

        # find the first two layers of autoencoder, and second two layers of regular network
        firstTwoLayers = currentAutoencoder[currentAutoencoder['nodeLayer'] < 2]
        lastTwoLayers = currentNetwork[currentNetwork['nodeLayer'] > 1]

        # concat the two sections together
        self.network = pd.concat([firstTwoLayers, lastTwoLayers], axis=0)

    def forwardPass(self, currentInputBatch, returnTestSet=False, isAutoEncoder=False):
        '''
        The estimation of output generation based on the current weights of our neural network

        @param currentInputBatch: the record that we wish to test
        @param returnTestSet: whether to return our test set output or not
        @param isAutoEncoder: whether or not we are training an autoencoder
        @return:
        '''
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

            # init values to be used for autoencoder training
            totalEstimatedOutput = []
            totalLossValue = []

            # update our input layers
            inputIndex = self.network[self.network['nodeLayer'] == 0].index.tolist()
            for currentIndex in inputIndex:
                currentName = self.network.loc[currentIndex, 'nodeName']
                self.network.loc[currentIndex, 'currentOutputs'].append(currentInputRecord[currentName].iloc[0])

            # iterate through all the layers within our network
            for currentLayer in layerIndices:
                # find the nodes relevant to that layer
                currentLayerNodes = self.network[self.network['nodeLayer'] == currentLayer]

                # grab previous layer outputs as the last item in their store list of outputs
                prevLayerOutputs = self.network[self.network['nodeLayer'] == currentLayer - 1][['nodeName', 'currentOutputs']].set_index('nodeName')
                prevLayerOutputs['currentOutputs'] = prevLayerOutputs['currentOutputs'].apply(lambda x: x[-1])

                # add bias input term
                prevLayerOutputs.loc['Intercept'] = 1

                # iterate through all the nodes in that layer
                for currentNode in currentLayerNodes.index.tolist():
                    # grab the current weight map
                    weightMap = self.network.loc[currentNode, 'weightMap']

                    # convert the weight map to a dataframe
                    dictionaryTable = pd.DataFrame.from_dict(weightMap, orient='index')

                    # combine our weight map to our input values
                    mappingTable = pd.concat([prevLayerOutputs, dictionaryTable], axis=1)
                    mappingTable.columns = ['Inputs', 'Weights']

                    # find our node input as the sum product of previous layer and weights
                    inputValue = (mappingTable['Inputs'] * mappingTable['Weights']).sum()

                    # store this input
                    self.network.loc[currentNode, 'currentInputs'].append(inputValue)

                    # if output layer, add in our estimate and actual outputs
                    if currentLayer == self.network['nodeLayer'].max():
                        # if autoencoder, compare our value to the relevant input
                        if isAutoEncoder:
                            # if
                            self.network.loc[currentNode, 'currentOutputs'].append(inputValue)
                            nodeName = self.network.loc[currentNode, 'nodeName']
                            inputComparisonValue = currentInputRecord[nodeName].iloc[0]
                            self.network.loc[currentNode, 'actualValue'].append(inputComparisonValue)

                            # add our input to the network record
                            totalEstimatedOutput.append(inputValue)
                            totalLossValue.append((inputValue - inputComparisonValue) ** 2)

                        # if reg, add the estimated value and class value
                        elif self.isReg:
                            # add our input and actual value for later comparison
                            self.network.loc[currentNode, 'currentOutputs'].append(inputValue)
                            self.network.loc[currentNode, 'actualValue'].append(currentClass)

                            # store our output value and loss to our test set
                            currentInputBatch.loc[currentInputID, 'estimatedOutput'] = inputValue
                            currentInputBatch.loc[currentInputID, 'lossValue'] = (inputValue - currentClass)**2

                        # otherwise add our target output and softmax numerator to be normalized after node processing complete
                        else:
                            softmaxOutputs[currentNode] = math.exp(inputValue)
                            self.network.loc[currentNode, 'actualValue'].append(int(currentClass == self.network.loc[currentNode, 'nodeName']))

                    # if not an output node, then we need to take our sigmoid function
                    else:
                        self.network.loc[currentNode, 'currentOutputs'].append(1 / (1 + math.exp(-inputValue)))

                # add our normalized softmax values as outputs
                if (not self.isReg) and currentLayer == self.network['nodeLayer'].max() and not isAutoEncoder:
                    # find denom of softmax function
                    totalSoftmax = sum(softmaxOutputs.values())

                    # find our adjusted softmax relative to the entire suite of outputs
                    for nodeID, softmaxValue in softmaxOutputs.items():
                        softmaxValue = softmaxValue / totalSoftmax
                        self.network.loc[nodeID, 'currentOutputs'].append(softmaxValue)

                        # if the current output is the correct one, then take cross entropy loss
                        if self.network.loc[nodeID, 'actualValue'][-1] == 1:
                            estimatedValue = self.network.loc[max(softmaxOutputs, key=softmaxOutputs.get), 'nodeName']
                            currentInputBatch.loc[currentInputID, 'estimatedOutput'] = estimatedValue
                            crossEntropyLoss = -1 * np.log10(softmaxValue)
                            currentInputBatch.loc[currentInputID, 'lossValue'] = crossEntropyLoss

                # for autoencoder, we take the mean of our estimated output/loss
                elif isAutoEncoder and currentLayer == self.network['nodeLayer'].max():
                    currentInputBatch.loc[currentInputID, 'estimatedOutput'] = sum(totalEstimatedOutput)/len(totalEstimatedOutput)
                    currentInputBatch.loc[currentInputID, 'lossValue'] = sum(totalLossValue)/len(totalLossValue)

        # if we are returning the test set values, then we need to drop our test values for next run
        if returnTestSet:
            self.dropLastForwardPass()
            return currentInputBatch

    def updatePartialErrors(self, indexStoppingPoint=0):
        """
        Allow us to iterate back through our network to add our partial errors at each node

        @param indexStoppingPoint: the layer that we want to stop updating at
        @return: nothing
        """
        # iterate through our network backwards
        currentLayer = self.network['nodeLayer'].max()

        # check layer to stop updating at
        while currentLayer > indexStoppingPoint:
            # find our current layer of nodes
            currentLayerNodes = self.network[self.network['nodeLayer'] == currentLayer]
            nextLayerNodes = self.network[self.network['nodeLayer'] == currentLayer + 1]

            # iterate through our current layer nodes
            for currentNode in currentLayerNodes.index.tolist():
                # start with the last layer
                if currentLayer == self.network['nodeLayer'].max():
                    # grab our list of test cases to compare actual versus estimated to get mean difference
                    currentActualList = np.array(currentLayerNodes.loc[currentNode, 'actualValue'])
                    currentOutputList = np.array(currentLayerNodes.loc[currentNode, 'currentOutputs'])
                    outputError = currentActualList - currentOutputList
                    self.network.loc[currentNode, 'currentPartialError'].extend(outputError.tolist())

                # if not last layer, then need to find weighted error relative to its weights to its next layer
                else:
                    # grab the current node's name
                    currentNodeName = currentLayerNodes.loc[currentNode, 'nodeName']
                    nextLayerWeightedError = 0

                    # add up all the weighted errors of next layer
                    for currentNextNode in nextLayerNodes.index.tolist():
                        currentNextWeightMap = nextLayerNodes.loc[currentNextNode, 'weightMap']
                        nextLayerWeightedError += currentNextWeightMap[currentNodeName] * np.array(nextLayerNodes.loc[currentNextNode, 'currentPartialError'])
                    self.network.loc[currentNode, 'currentPartialError'].extend(nextLayerWeightedError.tolist())

            # move back one layer
            currentLayer -= 1

    def updateWeights(self, currentLearningRate, indexStoppingPoint=0):
        """
        Update each weight based on the relevant gradient descent formula

        @param currentLearningRate: the learning rate that helps the gradient descent
        @param indexStoppingPoint: the layer index to stop updating at
        @return: nothing
        """

        # iterate through our network backwards
        currentLayer = self.network['nodeLayer'].max()
        while currentLayer > indexStoppingPoint:
            # iterate through our current layer nodes
            currentLayerNodes = self.network[self.network['nodeLayer'] == currentLayer]
            for currentNode in currentLayerNodes.index.tolist():
                # grab relevant weight map and partial error to help update our weights
                currentWeightMap = self.network.loc[currentNode, 'weightMap']
                currentPartialError = np.array(currentLayerNodes.loc[currentNode, 'currentPartialError'])

                if currentLayer == self.network['nodeLayer'].max():
                    currentPartialDeriv = 1
                else:
                    currentNodeOutputs = np.array(self.network.loc[currentNode, 'currentOutputs'])
                    currentPartialDeriv = currentNodeOutputs * (1 - currentNodeOutputs)

                # iterate through the weights to update by learning rate and current partial error
                for inputName, inputWeight in currentWeightMap.items():
                    # find the previous layer outputs correcting manually for intercept to get our mean adjustment value
                    if inputName == 'Intercept':
                        averageAdjustment = np.mean(currentPartialError * currentPartialDeriv)
                    else:
                        previousLayerNodeID = self.network[self.network['nodeName'] == inputName].index[0]
                        previousLayerOutput = np.array(self.network.loc[previousLayerNodeID, 'currentOutputs'])
                        averageAdjustment = np.mean(currentPartialError * currentPartialDeriv * previousLayerOutput)
                    # adjust the weight by this calculated value
                    currentWeightMap[inputName] = inputWeight + currentLearningRate * averageAdjustment

            # move back one layer
            currentLayer -= 1

        # reset out output values once our weights have been updated
        self.dropLastForwardPass()

    def dropLastForwardPass(self):
        """
        remove the test results from previous pass if no longer needed

        @return: nothing
        """
        # reset out output values once our weights have been updated
        self.network['currentInputs'] = self.network['currentInputs'].apply(lambda x: [])
        self.network['currentOutputs'] = self.network['currentOutputs'].apply(lambda x: [])
        self.network['actualValue'] = self.network['actualValue'].apply(lambda x: [])
        self.network['currentPartialError'] = self.network['currentPartialError'].apply(lambda x: [])

    def trainNetwork(self, tuneSet, learningRate, indexStop=0, isTune=False, isAutoEncoder=False):
        """
        train the network on the provided test set

        @param tuneSet: the set that we can help to test the trained network for convergence
        @param learningRate: the rate used in the gradient descent function
        @param indexStop: the index for which we want to stop updating on
        @param isTune: whether this training is meant for tuning hyperparameters
        @param isAutoEncoder: whether this training is meant for an autoencoder
        @return: nothing
        """

        # variable to test if we should continue to train the network
        keepRunning = True

        # the number of runs before a check for new min
        numRunsBeforeCheck = 3

        # run initial check as measuring point
        print("\n")
        print("**Starting initial check for " + self.dataName + " " + datetime.now().strftime("%d.%m.%Y_%I.%M.%S") + "**")
        initOutput = self.forwardPass(tuneSet, returnTestSet=True, isAutoEncoder=isAutoEncoder)
        initOutputLoss = initOutput['lossValue'].mean()
        print("Our initial loss is: " + str(initOutputLoss))
        print("\n")

        # init current min at initial loss metric
        currentMin = initOutputLoss
        totalRuns = 1

        # run until we reach some measure of convergence explained below
        while keepRunning:

            # run a new batch of tests
            for index in range(numRunsBeforeCheck):

                # copy over our training data to be slowly batched in our training
                remTestSet = self.trainData.copy()
                # set the size of our samples
                amountToSample = int(.02 * len(remTestSet))

                print("**Starting training " + str(totalRuns) + " for " + self.dataName + " " + datetime.now().strftime("%d.%m.%Y_%I.%M.%S") + "**")
                # run until we run out of test cases to try
                while len(remTestSet) > 0:
                    # adjust the amount to sample to ensure we don't sample more than is there
                    if amountToSample > len(remTestSet):
                        amountToSample = len(remTestSet)

                    # adjust the learning rate for the size of the test sample
                    adjLearningRate = learningRate * amountToSample

                    # sample the desired amount, remove those from remaining set
                    currentTestSet = remTestSet.sample(amountToSample)
                    remTestSet = remTestSet.drop(currentTestSet.index)

                    # run the forward pass of our algo
                    self.forwardPass(currentTestSet,  returnTestSet=False, isAutoEncoder=isAutoEncoder)

                    # update our partial errors
                    self.updatePartialErrors()

                    # update our weights
                    self.updateWeights(adjLearningRate, indexStop)

                # test against our validation set
                print("**Starting validation check for " + self.dataName + " " + datetime.now().strftime("%d.%m.%Y_%I.%M.%S") + "**")
                validationOutput = self.forwardPass(tuneSet, returnTestSet=True, isAutoEncoder=isAutoEncoder)
                validationLossRate = validationOutput['lossValue'].mean()
                print("Our output is: " + str(validationLossRate))

                # save down a record of this run
                networkCopy = aux.hardCopyDataframe(self.network)
                self.trainNetworkList.append(networkCopy)
                self.trainPerformanceList.append(validationLossRate)

                # increment our test counter
                totalRuns += 1

            # save down our current min and find new min
            prevMin = currentMin
            currentMin = min(self.trainPerformanceList)

            # check for convergence based on whether a new min is found and the size of the change
            sameMin = currentMin == prevMin
            minChange = (prevMin - currentMin)/prevMin

            # check for convergence.
            # tuning requires less runs because we are just testing for efficacy rather than truly training a network
            if isTune:
                convergenceTest = sameMin or minChange < .01 or totalRuns >= 15
            else:
                convergenceTest = sameMin or minChange < .02 or totalRuns >= 30

            # check for convergence
            if convergenceTest:
                keepRunning = False

            print("\n")

        # restore the network
        networkToKeep = self.trainNetworkList[self.trainPerformanceList.index(currentMin)]
        self.network = aux.hardCopyDataframe(networkToKeep)

        # reset our train lists
        self.trainNetworkList = []
        self.trainPerformanceList = []
