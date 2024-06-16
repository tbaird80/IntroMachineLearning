import pandas as pd
from datetime import datetime
import AuxML1

def KNNTest(dataSetID, features, targets, normalCol, tuningMap, hybridCols, isReg):
    # create file to store output
    validationTestFileName = dataSetID + "/CrossValidationTestFile.csv"

    #grab our tuned variables
    tunedK = tuningMap['k'].iloc[0]
    tunedP = tuningMap['p'].iloc[0]
    tunedE = tuningMap['e'].iloc[0]
    tunedS = tuningMap['s'].iloc[0]

    #test
    currentTest = 1
    kNearestOutput = pd.DataFrame()

    for loopIndex in range(5):
        trainSet1Raw, trainSet2Raw = AuxML1.splitDataFrame(features, targets, .5, isReg)

        crossSetTrain1 = trainSet1Raw.copy()
        crossSetTrain2 = trainSet2Raw.copy()
        crossSetTest1 = trainSet1Raw.copy()
        crossSetTest2 = trainSet2Raw.copy()

        crossSetTrain1 = AuxML1.normalizeNumberValues(crossSetTrain1, trainSet1Raw, normalCol)
        crossSetTrain2 = AuxML1.normalizeNumberValues(crossSetTrain2, trainSet2Raw, normalCol)

        # normalize using their respective train set
        crossSetTest1 = AuxML1.normalizeNumberValues(crossSetTest1, trainSet2Raw, normalCol)
        crossSetTest2 = AuxML1.normalizeNumberValues(crossSetTest2, trainSet1Raw, normalCol)

        print("-----------------Starting the test set " + str(loopIndex) + " " + datetime.now().strftime("%d.%m.%Y_%I.%M.%S") + "-------------------------")
        print("***Condensing Train Set 1 " + datetime.now().strftime("%d.%m.%Y_%I.%M.%S") + "***")
        crossSetTrain1Condensed = AuxML1.condensedNearestNeighbor(crossSetTrain1, targets, 1, tunedP, hybridCols, tunedE, tunedS, isReg)
        print("***Condensing Train Set 2 " + datetime.now().strftime("%d.%m.%Y_%I.%M.%S") + "***")
        crossSetTrain2Condensed = AuxML1.condensedNearestNeighbor(crossSetTrain2, targets, 1, tunedP, hybridCols, tunedE, tunedS, isReg)

        kNearestTest = pd.DataFrame({
            'testID': [],
            'nearestNeighbors': [],
            'expectedValue': [],
            'actualValue': [],
            'correctAssignment': []
        })

        kNearestTest['expectedValue'] = kNearestTest['expectedValue'].astype('object')

        print("***Testing Set 1 Against Condensed Set 2 " + datetime.now().strftime("%d.%m.%Y_%I.%M.%S") + "***")
        for currentTestRow1 in crossSetTest1.index.tolist():
            kNearestTest.loc[currentTestRow1] = AuxML1.kNearestNeighbor(crossSetTest1.loc[currentTestRow1],
                                                                        crossSetTrain2Condensed,
                                                                        targets,
                                                                        tunedK,
                                                                        tunedP,
                                                                        hybridCols,
                                                                        tunedE,
                                                                        tunedS,
                                                                        isReg)


        # update test ID and add our output to our logs
        kNearestTest['testID'] = currentTest
        kNearestOutput = pd.concat([kNearestOutput, kNearestTest])
        kNearestOutput.to_csv(validationTestFileName, index=True)

        # reset our table once tracked above before moving to the next index
        kNearestTest = kNearestTest.drop(kNearestTest.index.to_list())

        print("***Testing Set 2 Against Condensed Set 1 " + datetime.now().strftime("%d.%m.%Y_%I.%M.%S") + "***")
        for currentTestRow2 in crossSetTest2.index.tolist():
            kNearestTest.loc[currentTestRow2] = AuxML1.kNearestNeighbor(crossSetTest2.loc[currentTestRow2],
                                                                        crossSetTrain1Condensed,
                                                                        targets,
                                                                        tunedK,
                                                                        tunedP,
                                                                        hybridCols,
                                                                        tunedE,
                                                                        tunedS,
                                                                        isReg)

        # update test ID and add our output to our logs
        kNearestTest['testID'] = currentTest
        kNearestOutput = pd.concat([kNearestOutput, kNearestTest])
        kNearestOutput.to_csv(validationTestFileName, index=True)

        # increment our test case
        currentTest += 1

    # find the average accuracy of the set of tests and print to console
    averageAccuracy = AuxML1.testEffectiveness(kNearestOutput, isReg)
    print("The average accuracy of this test is: " + str(averageAccuracy))

    #print out our null test result as a as a
    if isReg:
        nullModel = features.join(targets)
        nullModel['expectedValue'] = nullModel['Class'].mean()
        MSE = 1/len(nullModel) * sum((nullModel['Class'] - nullModel['expectedValue'])**2)
        print("The average accuracy of the null model is: " + str(MSE))

    else:
        nullModel = features.join(targets)
        nullModel['expectedValue'] = nullModel['Class'].value_counts().idxmax()
        nullModel['correctAssignment'] = nullModel['expectedValue'] == nullModel['Class']

        print("The average accuracy of the null model is: " + str(nullModel['correctAssignment'].mean()))
