import pandas as pd
import AuxML1

def KNNTest(dataSetTitle, trainSetRaw, targets, normalCol, tuningMap, hybridCols, isReg):
    # Get the current timestamp
    currentTimestamp = datetime.now()
    timestampStr = currentTimestamp.strftime("%d.%m.%Y_%I.%M.%S")

    fileName = dataSetTitle + "TestFile" + timestampStr + ".csv"

    tunedK = tuningMap['k']
    tunedP = tuningMap['p']
    tunedE = tuningMap['e']
    tunedS = tuningMap['s']

    #test
    averageAccuracy = 0

    for loopIndex in range(5):
        trainSet1Raw, trainSet2Raw = AuxML1.splitDataFrame(trainSetRaw, targets, .5)

        crossSetTrain1 = trainSet1Raw.copy()
        crossSetTrain2 = trainSet2Raw.copy()
        crossSetTest1 = trainSet1Raw.copy()
        crossSetTest2 = trainSet2Raw.copy()

        crossSetTrain1 = AuxML1.normalizeNumberValues(crossSetTrain1, trainSet1Raw, normalCol)
        crossSetTrain2 = AuxML1.normalizeNumberValues(crossSetTrain2, trainSet2Raw, normalCol)

        #normalize using their respective train set
        crossSetTest1 = AuxML1.normalizeNumberValues(crossSetTest1, trainSet2Raw, normalCol)
        crossSetTest2 = AuxML1.normalizeNumberValues(crossSetTest2, trainSet1Raw, normalCol)

        print("-----------------Starting the test condensed Neighbor " + loopIndex + " -------------------------")
        print("***Test Set 1***")
        crossSetTrain1Condensed = AuxML1.condensedNearestNeighbor(crossSetTrain1, targets, 1, tunedP, hybridCols, tunedE, tunedS)
        print("***Test Set 2***")
        crossSetTrain2Condensed = AuxML1.condensedNearestNeighbor(crossSetTrain2, targets, 1, tunedP, hybridCols, tunedE, tunedS)

        print("***Condensed Test Set 1***")
        print(crossSetTrain1Condensed)
        print("***Condensed Test Set 2***")
        print(crossSetTrain2Condensed)

        kNearestTest = pd.DataFrame({
            'nearestNeighbors': [],
            'expectedValue': [],
            'actualValue': [],
            'correctAssignment': []
        })

        kNearestTest['expectedValue'] = kNearestTest['expectedValue'].astype('object')

        testRows1 = crossSetTest1.index.tolist()
        testRows2 = crossSetTest2.index.tolist()

        for currentTestRow1 in testRows1:
            kNearestTest.loc[currentTestRow1] = AuxML1.kNearestNeighbor(crossSetTest1.loc[currentTestRow1],
                                                                        crossSetTrain1Condensed,
                                                                        targets,
                                                                        tunedK,
                                                                        tunedP,
                                                                        hybridCols,
                                                                        tunedE,
                                                                        tunedS,
                                                                        isReg)

        averageAccuracy += AuxML1.testEffectiveness(kNearestTest, isReg)

        for currentTestRow2 in testRows2:
            kNearestTest.loc[currentTestRow2] = AuxML1.kNearestNeighbor(crossSetTest2.loc[currentTestRow2],
                                                                        crossSetTrain2Condensed,
                                                                        targets,
                                                                        tunedK,
                                                                        tunedP,
                                                                        hybridCols,
                                                                        tunedE,
                                                                        tunedS,
                                                                        isReg)

        averageAccuracy += AuxML1.testEffectiveness(kNearestTest, isReg)

    print("The average accuracy of this test is: " + averageAccuracy)

    if isReg:
        print("The average accuracy of the null model is: " + targets['Class'].mean())
    else:
        nullModel = features.join(targets)
        nullModel['expectedValue'] = targets['Class'].value_counts().idxmax()
        nullModel['correctAssignment'] = nullModel['expectedValue'] == nullModel['Class']

        print("The average accuracy of the null model is: " + nullModel['correctAssignment'].mean())