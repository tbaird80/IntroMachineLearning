import DataML1
import KNNMainML1

if __name__ == '__main__':
    # function inputs
    # grab data
    features, targets = DataML1.dataSourcing('ComputerHardware')
    # define the columns that need to be normalized
    normalCol = ['MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX']
    # define the features to be tuned
    tuningMap = {'p': [1, 2], 'k': [2, 3, 4, 5], 'e': [1], 'l': [1]}
    # define whether there are hybrid columns in that there are multiple data types to worry about
    hybridCols = True
    # define whether it is a regression
    regression = True

    KNNMainML1.KNNMain(features, targets, normalCol, tuningMap, hybridCols, regression)