import DataML1
import KNNMTuningML1
#import KNNTestML1

if __name__ == '__main__':
    # function inputs
    # data set title
    dataTitle = 'BreastCancer'

    # grab data
    features, targets = DataML1.dataSourcing(dataTitle)

    # define the columns that need to be normalized
    normalCol = features.columns

    # define the features to be tuned
    tuningMap = {'p': [1, 2], 'k': [2, 3, 4, 5], 'e': [1], 's': [1]}

    # define whether there are hybrid columns in that there are multiple data types to worry about
    hybridCols = False

    # define whether it is a regression
    regression = False

    tuneParameterOutput, testFeatures = KNNTuningML1.KNNTuning(dataTitle, features, targets, normalCol, tuningMap, hybridCols, regression)

    print(tuneParameterOutput)

    #maxTune = tuneParameterOutput.nlargest(1, 'AveragePerformance')

    #KNNTest.KNNTest(dataTitle, testFeatures, targets, normalCol, maxTune, hybridCols, regression)

