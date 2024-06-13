import DataML1
from datetime import datetime
import KNNTuningML1
import KNNTestML1
import os
import pandas as pd

if __name__ == '__main__':
    # function inputs
    # data set title
    dataTitle = 'ComputerHardware'

    # grab data
    features, targets = DataML1.dataSourcing(dataTitle)

    # check summary of the targets
    featuresSummary = features.describe()
    targetsSummary = targets.describe()

    # # define the columns that need to be normalized
    # normalCol = ['MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX']
    #
    # # define the features to be tuned
    # tuningMap = {'p': [1, 2], 'k': [2, 3, 4, 5], 'e': [1], 'l': [1]}
    #
    # # define whether there are hybrid columns in that there are multiple data types to worry about
    # hybridCols = True
    #
    # # define whether it is a regression
    # regression = True
    #
    # # ----------------------Tuning-----------------------------
    # # Get the current timestamp and create our own unique new directory
    # currentTimestamp = datetime.now()
    # timestampStr = currentTimestamp.strftime("%d.%m.%Y_%I.%M.%S")
    # uniqueTestID = dataTitle + "/" + timestampStr
    # os.makedirs(uniqueTestID)
    #
    # # shrink test set for testing
    # testSize = round(len(features) * .1)
    # features = features.sample(n=testSize)
    # targets = targets.loc[features.index.tolist()]
    #
    # # tune our parameters
    # tuneParameterOutput, testFeatures = KNNTuningML1.KNNTuning(uniqueTestID, features, targets, normalCol, tuningMap, hybridCols, regression)
    # print(tuneParameterOutput.nlargest(1, 'AveragePerformance'))

    # ----------------------Testing-----------------------------
    # source our data
    # uniqueTestID = dataTitle + '/'
    #
    # #read in our files from tuning
    # tunedParametersFile = uniqueTestID + "/ParameterTuningFile.csv"
    # tuningTestSet = uniqueTestID + "/TestSetRecord.csv"
    # tuneParameterOutput = pd.read_csv(tunedParametersFile, index_col=0)
    # tuneTestFeatures = pd.read_csv(tuningTestSet, index_col=0)
    #
    # maxTune = tuneParameterOutput.nlargest(1, 'AveragePerformance')
    # testFeatures = features.loc[tuneTestFeatures.index.tolist()]
    #
    # # shrink test set for testing
    # # testSize = round(len(testFeatures) * .1)
    # # testFeatures = testFeatures.sample(n=testSize)
    # # targets = targets.loc[testFeatures.index.tolist()]
    #
    # KNNTestML1.KNNTest(uniqueTestID, testFeatures, targets, normalCol, maxTune, hybridCols, regression)
