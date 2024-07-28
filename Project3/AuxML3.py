from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
import os
from datetime import datetime
import NeuralNetworkClass as network
import copy

def runTest(dataSetName, fullDataSet, isReg, normalCol, networkType, learningRate=0, numHiddenNodesPercentage=0, isTune=False):

    # create current test directory
    uniqueTestDir = dataSetName + "/" + networkType
    if isTune:
        batchesToRun = 1
    else:
        uniqueTestDir = uniqueTestDir + "/" + datetime.now().strftime("%d.%m.%Y_%I.%M.%S")
        getDirectory(uniqueTestDir)
        currentTestFile = uniqueTestDir + "/testOutput.csv"
        batchesToRun = 5
        learningRate, numHiddenNodesPercentage = getTunedParameters(dataSetName, networkType)

    currentIndexStop = 0

    # create test table
    currentOutput = pd.DataFrame()

    # we want to run a 5x2, so create 5 loops
    currentTestBatch = 1
    currentTestID = 1
    while currentTestBatch <= batchesToRun:
        tuneSet, testSet1, testSet2 = createTuneTrainTest(fullDataSet, isReg, normalCol)

        currentTestRound = 1
        while currentTestRound <= 2:
            if currentTestRound == 1:
                currentTrain = testSet1.copy()
                currentTest = testSet2.copy()
            else:
                currentTrain = testSet2.copy()
                currentTest = testSet1.copy()

            currentNetwork = network.NNet(dataSetName=dataSetName, isRegression=isReg, trainingData=currentTrain,
                                          normalCols=normalCol, proportionHiddenNodesToInput=numHiddenNodesPercentage, networkType=networkType)

            print("\n")
            print("*******************************************************************************************************************")
            print("*************************************Next Training Run*************************************************************")
            print("*******************************************************************************************************************")
            print("***********************Starting testing network " + dataSetName + " " + str(currentTestID) +
                  " " + datetime.now().strftime("%d.%m.%Y_%I.%M.%S") + "*****************************")

            currentNetwork.trainNetwork(tuneSet, learningRate, indexStop=currentIndexStop, isTune=isTune)

            if isTune:
                print("\n")
                print("**Checking performance for tuned hyperparameter " + currentNetwork.dataName + " " + str(currentTestID) + " "
                      + datetime.now().strftime("%d.%m.%Y_%I.%M.%S") + "**")
                validationOutput = currentNetwork.forwardPass(tuneSet, returnTestSet=True)
                validationLossRate = validationOutput['lossValue'].mean()
                print(validationLossRate)

                validationOutput['testID'] = currentTestID
                validationOutput['learningRate'] = learningRate
                validationOutput['numHiddenNodesPercentage'] = numHiddenNodesPercentage

            else:
                print("\n")
                print("**Testing tuned network " + currentNetwork.dataName + " " + str(currentTestID) + " "
                      + datetime.now().strftime("%d.%m.%Y_%I.%M.%S") + "**")
                validationOutput = currentNetwork.forwardPass(currentTest, returnTestSet=True)
                validationLossRate = validationOutput['lossValue'].mean()
                print(validationLossRate)

                validationOutput['testID'] = currentTestID
                validationOutput['learningRate'] = learningRate
                validationOutput['numHiddenNodesPercentage'] = numHiddenNodesPercentage

            currentOutput = pd.concat([currentOutput, validationOutput])

            if not isTune:
                currentOutput.to_csv(currentTestFile, index=True)

            currentTestRound += 1
            currentTestID += 1
        currentTestBatch += 1

    return uniqueTestDir, currentOutput

def tuneNetwork(dataSetName, fullDataSet, isReg, normalCol, networkType):

    # hyperparameter options
    if dataSetName == 'Abalone':
        learningRateOptions = [.001, .01, .05]
    else:
        learningRateOptions = [.001, .01, .1]
    numHiddenNodesPercentageOptions = [.5, .75]

    if networkType == 'Simple':
        for currentLearnRate in learningRateOptions:
            print("\n")
            print("-------------------------------------------------------------------------------------------------------------------")
            print("-------------------------------------Next Tuning Run---------------------------------------------------------------")
            print("-------------------------------------------------------------------------------------------------------------------")
            print("---------------------------------Starting tuning learning rate " + dataSetName + " " + str(currentLearnRate) +
                      " " + datetime.now().strftime("%d.%m.%Y_%I.%M.%S") + "------------")

            testDir, testOutput = runTest(dataSetName, fullDataSet, isReg, normalCol, networkType, learningRate=currentLearnRate, numHiddenNodesPercentage=.5, isTune=True)

            # create test directory if not present
            testDir += "/TuningTests"
            getDirectory(testDir)
            # create test file path
            currentTestFile = testDir + "/" + str(currentLearnRate) + "testOutput.csv"

            testOutput.to_csv(currentTestFile, index=True)

    else:
        # set all of our options
        parameterOptions = pd.DataFrame({'learningRate': learningRateOptions * 2,
                                         'numHiddenNodes': numHiddenNodesPercentageOptions * 3})

        parameterOptions = parameterOptions.sort_values(by='learningRate')

        # iterate through all the parameter options
        for currentParametersIndex in parameterOptions.index.tolist():
            currentLearningRate = parameterOptions.loc[currentParametersIndex, 'learningRate']
            currentHiddenNodePercent = parameterOptions.loc[currentParametersIndex, 'numHiddenNodes']

            print("-------------------------------------------------------------------------------------------------------------------")
            print("-------------------------------------Next Tuning Run---------------------------------------------------------------")
            print("-------------------------------------------------------------------------------------------------------------------")
            print("---------------------------------Starting tuning learning rate " + dataSetName + " " + str(currentLearningRate) + " and percent of " + str(currentHiddenNodePercent) +
                  " " + datetime.now().strftime("%d.%m.%Y_%I.%M.%S") + "------------")

            testDir, testOutput = runTest(dataSetName, fullDataSet, isReg, normalCol, networkType, learningRate=currentLearningRate, numHiddenNodesPercentage=currentHiddenNodePercent, isTune=True)

            # create test directory if not present
            testDir += "/TuningTests"
            getDirectory(testDir)
            # create test file path
            currentTestFile = testDir + "/" + str(currentLearningRate) + "LearningRate" + str(currentHiddenNodePercent) + "NodePercent" + "testOutput.csv"

            testOutput.to_csv(currentTestFile, index=True)

    print("-------------------------------------------------------------------------------------------------------------------")
    print("-------------------------------------Done Tuning Learning Rate-----------------------------------------------------")
    print("-------------------------------------------------------------------------------------------------------------------")
    print("---------------------------------Finished tuning learning rate " + dataSetName + " " + datetime.now().strftime("%d.%m.%Y_%I.%M.%S") + "-------------------")

def runWithAutoEncoder(dataSetName, fullDataSet, isReg, normalCol, networkType='AutoEncoder'):
    # create current test directory
    uniqueTestDir = dataSetName + "/" + networkType + "/" + datetime.now().strftime("%d.%m.%Y_%I.%M.%S")
    getDirectory(uniqueTestDir)

    currentAutoEncoderFile = uniqueTestDir + "/autoEncoders.csv"
    currentAutoEncoderTestFile = uniqueTestDir + "/autoEncodersTest.csv"
    currentTestFile = uniqueTestDir + "/testOutput.csv"
    batchesToRun = 5
    learningRate, numHiddenNodesPercentage = getTunedParameters(dataSetName, testType='BackPro')

    # create test table
    currentAutoEncoderRun = pd.DataFrame()
    currentOutput = pd.DataFrame()
    currentAutoEncoder = pd.DataFrame()

    # we want to run a 5x2, so create 5 loops
    currentTestBatch = 1
    currentTestID = 1

    while currentTestBatch <= batchesToRun:
        tuneSet, testSet1, testSet2 = createTuneTrainTest(fullDataSet, isReg, normalCol)

        currentTestRound = 1
        while currentTestRound <= 2:
            if currentTestRound == 1:
                currentTrain = testSet1.copy()
                currentTest = testSet2.copy()
            else:
                currentTrain = testSet2.copy()
                currentTest = testSet1.copy()

            currentNetwork = network.NNet(dataSetName=dataSetName, isRegression=isReg, trainingData=currentTrain.copy(),
                                          normalCols=normalCol, proportionHiddenNodesToInput=numHiddenNodesPercentage, networkType='BackPro')

            tempIsReg = currentNetwork.isReg
            tempFullNetwork = hardCopyDataframe(currentNetwork.network)

            currentNetwork.isReg = True
            currentNetwork.network = hardCopyDataframe(currentNetwork.autoencoder)

            print("\n")
            print("*******************************************************************************************************************")
            print("*************************************Create Autoencoder************************************************************")
            print("*******************************************************************************************************************")
            print("***********************Starting testing network " + dataSetName + " " + str(currentTestID) +
                  " " + datetime.now().strftime("%d.%m.%Y_%I.%M.%S") + "*****************************")

            currentNetwork.trainNetwork(tuneSet.copy(), learningRate, indexStop=0, isAutoEncoder=True)

            print("\n")
            print("**Testing tuned autoencoder " + currentNetwork.dataName + " " + str(currentTestID) + " "
                  + datetime.now().strftime("%d.%m.%Y_%I.%M.%S") + "**")
            validationOutputTest = currentNetwork.forwardPass(currentTest.copy(), returnTestSet=True, isAutoEncoder=True)
            validationOutputTrain = currentNetwork.forwardPass(currentTrain.copy(), returnTestSet=True, isAutoEncoder=True)
            validationLossRateTest = validationOutputTest['lossValue'].mean()
            validationLossRateTrain = validationOutputTrain['lossValue'].mean()
            print("Test validation: " + str(validationLossRateTest))
            print("Train validation: " + str(validationLossRateTrain))

            # write current autoencoder test to memory
            validationOutputTest['testID'] = "Test" + str(currentTestID)
            validationOutputTrain['testID'] = "Train" + str(currentTestID)
            currentAutoEncoderRun = pd.concat([currentAutoEncoderRun, validationOutputTest, validationOutputTrain])
            currentAutoEncoderRun.to_csv(currentAutoEncoderTestFile, index=True)

            # reset our network values
            currentNetwork.isReg = tempIsReg
            currentNetwork.autoencoder = hardCopyDataframe(currentNetwork.network)
            currentNetwork.network = hardCopyDataframe(tempFullNetwork)
            currentNetwork.updateWithAutoencoder()

            # write current autoencoder to memory
            currentNetwork.autoencoder['testID'] = currentTestID
            currentAutoEncoder = pd.concat([currentAutoEncoder, currentNetwork.autoencoder])
            currentAutoEncoder.to_csv(currentAutoEncoderFile, index=True)

            print("\n")
            print("*******************************************************************************************************************")
            print("**********************************Post Autoencoder Training Run****************************************************")
            print("*******************************************************************************************************************")
            print("***********************Starting testing network " + dataSetName + " " + str(currentTestID) +
                  " " + datetime.now().strftime("%d.%m.%Y_%I.%M.%S") + "*****************************")

            currentNetwork.trainNetwork(tuneSet.copy(), learningRate, indexStop=1)

            print("\n")
            print("**Testing tuned network " + currentNetwork.dataName + " " + str(currentTestID) + " "
                  + datetime.now().strftime("%d.%m.%Y_%I.%M.%S") + "**")
            validationOutputTest = currentNetwork.forwardPass(currentTest.copy(), returnTestSet=True)
            validationOutputTrain = currentNetwork.forwardPass(currentTrain.copy(), returnTestSet=True)
            validationLossRateTest = validationOutputTest['lossValue'].mean()
            validationLossRateTrain = validationOutputTrain['lossValue'].mean()
            print("Test validation: " + str(validationLossRateTest))
            print("Train validation: " + str(validationLossRateTrain))

            # write current autoencoder test to memory
            validationOutputTest['testID'] = "Test" + str(currentTestID)
            validationOutputTrain['testID'] = "Train" + str(currentTestID)
            validationOutputTest['learningRate'] = learningRate
            validationOutputTrain['numHiddenNodesPercentage'] = numHiddenNodesPercentage
            validationOutputTest['learningRate'] = learningRate
            validationOutputTrain['numHiddenNodesPercentage'] = numHiddenNodesPercentage

            # write tests to csv
            currentOutput = pd.concat([currentOutput, validationOutputTest, validationOutputTrain])
            currentOutput.to_csv(currentTestFile, index=True)

            currentTestRound += 1
            currentTestID += 1
        currentTestBatch += 1

    return uniqueTestDir, currentOutput

def normalizeNumberValues(testSet, trainSet, colsNormalize):
    """
    Used to normalize values within data set where relevant

    @param testSet: the test set to be normalized
    @param trainSet: the train set for which the mean and std will be sourced from
    @param colsNormalize: the columns for which normalizing is needed within the test set
    @return: normalized dataframe
    """
    # iterate through our columns
    for currentCol in testSet.columns:
        # normalize the column if specified
        if currentCol in colsNormalize:
            dataMin = trainSet[currentCol].min()
            dataMax = trainSet[currentCol].max()
            testSet[currentCol] = (testSet[currentCol] - dataMin) / (dataMax - dataMin)
        else:
            continue
    #return testSet

def hardCopyDataframe(currentDataFrame):
    # need to adjust hard copy to account for dictionary columns
    dataframeCopy = currentDataFrame.applymap(lambda x: copy.deepcopy(x) if isinstance(x, dict) else x)
    return dataframeCopy

def getTunedParameters(dataName, testType):
    # Specify the directory containing the CSV files
    directory = dataName + '/' + testType + '/' + 'TuningTests'

    # List to hold DataFrames
    tuningOutputsList = []

    # Loop through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            # Read the CSV file
            filepath = directory + "/" + filename
            currentOutput = pd.read_csv(filepath)
            # Append the DataFrame to the list
            tuningOutputsList.append(currentOutput)

    # Concatenate all DataFrames in the list into one DataFrame
    tuningOutputs = pd.concat(tuningOutputsList, ignore_index=True)

    aggTuningOutputs = tuningOutputs[['learningRate', 'lossValue', 'numHiddenNodesPercentage']].groupby(['learningRate', 'numHiddenNodesPercentage'], as_index=False).mean()

    learningRate = aggTuningOutputs.loc[aggTuningOutputs['lossValue'].idxmin(), 'learningRate']
    numHiddenNodesPercentage = aggTuningOutputs.loc[aggTuningOutputs['lossValue'].idxmin(), 'numHiddenNodesPercentage']

    return float(learningRate), float(numHiddenNodesPercentage)

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

def createTuneTrainTest(dataSet, isReg, normalCols):
    # split into tune, test, and train sets
    tuneSet, trainTestSet = splitDataFrame(dataSet, .2, isReg)
    trainSet, testSet = splitDataFrame(trainTestSet, .5, isReg)

    normalizeDataSet = trainSet.copy()
    normalizeNumberValues(tuneSet, normalizeDataSet, normalCols)
    normalizeNumberValues(trainSet, normalizeDataSet, normalCols)
    normalizeNumberValues(testSet, normalizeDataSet, normalCols)

    return tuneSet, trainSet, testSet

def getDirectory(desiredPath):
    # Check if the directory exists
    if not os.path.exists(desiredPath):
        # If the directory does not exist, create it
        os.makedirs(desiredPath)
        print(f'Directory "{desiredPath}" created.')
    else:
        print(f'Directory "{desiredPath}" already exists.')
