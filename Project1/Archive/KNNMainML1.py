import pandas as pd
import AuxML1

def KNNMain(features, targets, normalCol, tuningMap, hybridCols, isReg):
    # tuning
    tuneSetRaw, trainSetRaw = AuxML1.splitDataFrame(features, targets, .2)

    # creates all combinations of our dictionary
    tuneMultiIndex = pd.MultiIndex.from_product(tuningMap.values(), names=tuningMap.keys())
    allTuneParameters = pd.DataFrame(index=tuneMultiIndex).reset_index()

    subsetTuneParameters = allTuneParameters.sample(n=3)

    subsetTuneParameters['AveragePerformance'] = [0] * len(subsetTuneParameters)

    for currentParameter in subsetTuneParameters.index.tolist():
        for loopIndex in range(5):

            trainSet1Raw, trainSet2Raw = AuxML1.splitDataFrame(trainSetRaw, targets, .5)

            trainSet1 = trainSet1Raw.copy()
            trainSet2 = trainSet2Raw.copy()

            tuneSet1 = tuneSetRaw.copy()
            tuneSet2 = tuneSetRaw.copy()

            trainSet1 = AuxML1.normalizeNumberValues(trainSet1, trainSet1Raw, normalCol)
            trainSet2 = AuxML1.normalizeNumberValues(trainSet2, trainSet2Raw, normalCol)

            tuneSet1 = AuxML1.normalizeNumberValues(tuneSet1, trainSet1Raw, normalCol)
            tuneSet2 = AuxML1.normalizeNumberValues(tuneSet2, trainSet2Raw, normalCol)

            k = subsetTuneParameters.loc[currentParameter, 'k']
            p = subsetTuneParameters.loc[currentParameter, 'p']
            e = subsetTuneParameters.loc[currentParameter, 'e']
            s = subsetTuneParameters.loc[currentParameter, 's']

            print("-----------------Starting the tune condensed Neighbor " + loopIndex + " -------------------------")
            print("***Test Set 1***")
            trainSet1Condensed = AuxML1.condensedNearestNeighbor(trainSet1, targets, 1, p, hybridCols, e, s, isReg)
            print("***Test Set 2***")
            trainSet2Condensed = AuxML1.condensedNearestNeighbor(trainSet2, targets, 1, p, hybridCols, e, s, isReg)

            print("***Condensed Test Set 1***")
            print(trainSet1Condensed)
            print("***Condensed Test Set 2***")
            print(trainSet2Condensed)

            kNearestTune = pd.DataFrame({
                'nearestNeighbors': [],
                'expectedValue': [],
                'actualValue': [],
                'correctAssignment': []
            })

            kNearestTune['expectedValue'] = kNearestTune['expectedValue'].astype('object')

            tuneRows1 = tuneSet1.index.tolist()
            tuneRows2 = tuneSet2.index.tolist()

            for currentTuneRow1 in tuneRows1:
                kNearestTune.loc[currentTuneRow1] = AuxML1.kNearestNeighbor(tuneSet1.loc[currentTuneRow1],
                                                                            trainSet1Condensed,
                                                                            targets,
                                                                            k,
                                                                            p,
                                                                            hybridCols,
                                                                            e,
                                                                            s,
                                                                            isReg)

            subsetTuneParameters.loc[currentParameter, 'AveragePerformance'] += AuxML1.testEffectiveness(kNearestTune,
                                                                                                         isReg)

            for currentTuneRow2 in tuneRows2:
                kNearestTune.loc[currentTuneRow2] = AuxML1.kNearestNeighbor(tuneSet2.loc[currentTuneRow2],
                                                                            trainSet2Condensed,
                                                                            targets,
                                                                            k,
                                                                            p,
                                                                            hybridCols,
                                                                            e,
                                                                            s,
                                                                            isReg)

            print("***Current Tune Output***")
            print(AuxML1.testEffectiveness(kNearestTune, isReg))
            print(kNearestTune)
            subsetTuneParameters.loc[currentParameter, 'AveragePerformance'] += AuxML1.testEffectiveness(kNearestTune,
                                                                                                         isReg)

    print(subsetTuneParameters)
    maxTune = subsetTuneParameters.nlargest(1, 'AveragePerformance')
    tunedK = maxTune['k']
    tunedP = maxTune['p']
    tunedE = maxTune['e']
    tunedS = maxTune['s']

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