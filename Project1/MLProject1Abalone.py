import DataML1
from datetime import datetime
import KNNTuningML1
import KNNTestML1
import os
import pandas as pd

if __name__ == '__main__':
    '''
    This is our main function for the Abalone test set. It will define, tune, and test the set to return an effectiveness
    value for the k nearest neighbor algorithm. If you would like to run yourself, I would recommend doing so in chunks. Tune first,
    then test.
    
    '''

    # function inputs
    # data set title
    dataTitle = 'Abalone'

    # grab data
    features, targets = DataML1.dataSourcing(dataTitle)

    # define the columns that need to be normalized
    normalCol = ['Length', 'Diameter', 'Height',
                 'Whole_weight', 'Shucked_weight', 'Viscera_weight',
                 'Shell_weight']

    # define the features to be tuned
    tuningMap = {'p': [1, 2], 'k': [30, 40, 50], 'e': [2, 3], 's': [1, 2]}

    # define whether there are hybrid columns in that there are multiple data types to worry about
    hybridCols = True

    # define whether it is a regression
    regression = True

    # ----------------------Tuning-----------------------------
    # Get the current timestamp and create our own unique new directory
    currentTimestamp = datetime.now()
    timestampStr = currentTimestamp.strftime("%d.%m.%Y_%I.%M.%S")
    uniqueTestID = dataTitle + "/" + timestampStr
    os.makedirs(uniqueTestID)

    # shrink test set for testing
    # testSize = round(len(features) * .01)
    # features = features.sample(n=testSize)
    # targets = targets.loc[features.index.tolist()]

    # tune our parameters
    tuneParameterOutput, testFeatures = KNNTuningML1.KNNTuning(uniqueTestID, features, targets, normalCol, tuningMap, hybridCols, regression)
    print(tuneParameterOutput.nsmallest(1, 'AveragePerformance'))

    # # ----------------------Testing-----------------------------
    # source our data
    uniqueTestID = dataTitle + '/14.06.2024_06.54.25'

    # read in our files from tuning
    tunedParametersFile = uniqueTestID + "/ParameterTuningFile.csv"
    tuningRawDataFile = uniqueTestID + "/TuneRawData.csv"
    tuneParameterOutput = pd.read_csv(tunedParametersFile, index_col=0)
    tuneFeatures = pd.read_csv(tuningRawDataFile, index_col=0)

    # in this test, we only had time to run one tuning given the length to run. therefore we took largest to get the one that finished
    maxTune = tuneParameterOutput.nlargest(1, 'AveragePerformance')
    # grab the same test cases as defined as the 80% used for tuning
    testFeaturesIndices = features.index.difference(tuneFeatures.index)
    testFeatures = features.loc[testFeaturesIndices]

    # shrink test set for testing
    # testSize = round(len(features) * .8)
    # testFeatures = testFeatures.sample(n=testSize)
    # targets = targets.loc[testFeatures.index.tolist()]

    # run the test given best hyperparameters
    KNNTestML1.KNNTest(uniqueTestID, testFeatures, targets, normalCol, maxTune, hybridCols, regression)
