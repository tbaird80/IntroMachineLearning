import numpy as np
import pandas as pd
import DataML1

if __name__ == '__main__':
    # source our data
    dataTitle = 'ComputerHardware'
    uniqueTestID = dataTitle + '/14.06.2024_01.32.17'

    # grab data
    features, targets = DataML1.dataSourcing(dataTitle)

    # read in our files from tuning
    crossValidationOutput = "ComputerHardware/14.06.2024_04.52.03/CrossValidationTestFile.csv"

    testOutput = pd.read_csv(crossValidationOutput, index_col=0)

    targetsTest = targets.loc[testOutput.index.tolist()]
    targetsTuneIndex = features.index.difference(targetsTest.index)
    targetsTune = targets.loc[targetsTuneIndex]

    print(targetsTest['Class'].mean())
    print(targetsTune['Class'].mean())
