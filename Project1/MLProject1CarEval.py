import DataML1
from datetime import datetime
import KNNTuningML1
import KNNTestML1
import os
import pandas as pd

if __name__ == '__main__':
    '''
    This is our main function for the Car Evaluation test set. It will define, tune, and test the set to return an effectiveness
    value for the k nearest neighbor algorithm. If you would like to run yourself, I would recommend doing so in chunks. Tune first,
    then test.

    '''
    # function inputs
    # data set title
    dataTitle = 'CarEval'

    # grab data
    features, targets = DataML1.dataSourcing(dataTitle)

    # define the columns that need to be normalized
    normalCol = []

    # define the features to be tuned
    tuningMap = {'p': [1, 2], 'k': [12, 14, 16], 'e': [1], 's': [1]}

    # define whether there are hybrid columns in that there are multiple data types to worry about
    hybridCols = True

    # define whether it is a regression
    regression = False

    # # ----------------------Tuning-----------------------------
    # # Get the current timestamp and create our own unique new directory
    # currentTimestamp = datetime.now()
    # timestampStr = currentTimestamp.strftime("%d.%m.%Y_%I.%M.%S")
    # uniqueTestID = dataTitle + "/" + timestampStr
    # os.makedirs(uniqueTestID)
    #
    # # # shrink test set for testing
    # # testSize = round(len(features) * .05)
    # # features = features.sample(n=testSize)
    # # targets = targets.loc[features.index.tolist()]
    #
    # # tune our parameters
    # tuneParameterOutput, testFeatures = KNNTuningML1.KNNTuning(uniqueTestID, features, targets, normalCol, tuningMap, hybridCols, regression)
    # print(tuneParameterOutput.nlargest(1, 'AveragePerformance'))

    # ----------------------Testing-----------------------------
    # source our data
    uniqueTestID = dataTitle + '/'

    # read in our files from tuning
    tunedParametersFile = uniqueTestID + "/ParameterTuningFile.csv"
    tuningTestSet = uniqueTestID + "/TestSetRecord.csv"
    tuneParameterOutput = pd.read_csv(tunedParametersFile, index_col=0)
    tuneTestFeatures = pd.read_csv(tuningTestSet, index_col=0)

    # grab the best performing hyperparameters. This is a classification, so it is the largest % accuracy
    maxTune = tuneParameterOutput.nlargest(1, 'AveragePerformance')
    # grab the same test cases as defined as the 80% used for tuning
    testFeatures = features.loc[tuneTestFeatures.index.tolist()]

    # shrink test set for testing
    # testSize = round(len(testFeatures) * .05)
    # testFeatures = testFeatures.sample(n=testSize)
    # targets = targets.loc[testFeatures.index.tolist()]

    # run the test given best hyperparameters
    KNNTestML1.KNNTest(uniqueTestID, testFeatures, targets, normalCol, maxTune, hybridCols, regression)
