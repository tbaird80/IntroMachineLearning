import pandas as pd
from datetime import datetime
import AuxML1

def KNNTuning(dataSetTitle, features, targets, normalCol, tuningMap, hybridCols, isReg):
    # Get the current timestamp
    currentTimestamp = datetime.now()
    timestampStr = currentTimestamp.strftime("%d.%m.%Y_%I.%M.%S")

    fileName = dataSetTitle + "ParameterTuningFile" + timestampStr + ".csv"

    # tuning
    tuneSetRaw, trainSetRaw = AuxML1.splitDataFrame(features, targets, .2)

    # creates all combinations of our dictionary
    tuneMultiIndex = pd.MultiIndex.from_product(tuningMap.values(), names=tuningMap.keys())
    allTuneParameters = pd.DataFrame(index=tuneMultiIndex).reset_index()

    subsetTuneParameters = allTuneParameters.sample(n=3)

    subsetTuneParameters['AveragePerformance'] = [0] * len(subsetTuneParameters)
    subsetTuneParameters['TestsRun'] = [0] * len(subsetTuneParameters)

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
            print("***Condensing Test Set 1***")
            trainSet1Condensed = AuxML1.condensedNearestNeighbor(trainSet1, targets, 1, p, hybridCols, e, s, isReg)
            print("***Condensing Test Set 2***")
            trainSet2Condensed = AuxML1.condensedNearestNeighbor(trainSet2, targets, 1, p, hybridCols, e, s, isReg)

            kNearestTune = pd.DataFrame({
                'nearestNeighbors': [],
                'expectedValue': [],
                'actualValue': [],
                'correctAssignment': []
            })

            kNearestTune['expectedValue'] = kNearestTune['expectedValue'].astype('object')

            tuneRows1 = tuneSet1.index.tolist()
            tuneRows2 = tuneSet2.index.tolist()

            print("***Starting Tune " + subsetTuneParameters[currentParameter, 'TestsRun'] + " for parameter set " + currentParameter)
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

            subsetTuneParameters.loc[currentParameter, 'AveragePerformance'] += AuxML1.testEffectiveness(kNearestTune, isReg)
            subsetTuneParameters.loc[currentParameter, 'TestsRun'] += 1

            subsetTuneParameters.to_csv(fileName, index=True)

            print("***Starting Tune " + subsetTuneParameters[currentParameter, 'TestsRun'] + " for parameter set " + currentParameter)

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

            subsetTuneParameters.loc[currentParameter, 'AveragePerformance'] += AuxML1.testEffectiveness(kNearestTune,isReg)
            subsetTuneParameters.loc[currentParameter, 'TestsRun'] += 1

            subsetTuneParameters.to_csv(fileName, index=True)

    return subsetTuneParameters, trainSetRaw
