import pandas as pd
from datetime import datetime
import AuxML1

def KNNTuning(dataSetID, features, targets, normalCol, tuningMap, hybridCols, isReg):
    """

    @param dataSetID:
    @param features:
    @param targets:
    @param normalCol:
    @param tuningMap:
    @param hybridCols:
    @param isReg:
    @return:
    """

    tuneFileName = dataSetID + "/ParameterTuningFile.csv"
    testCasesFileName = dataSetID + "/TestSetRecord.csv"
    tuneRawFileName = dataSetID + "/TuneRawData.csv"

    # tuning
    tuneSetRaw, trainSetRaw = AuxML1.splitDataFrame(features, targets, .2, isReg)

    # creates all combinations of our dictionary
    tuneMultiIndex = pd.MultiIndex.from_product(tuningMap.values(), names=tuningMap.keys())
    allTuneParameters = pd.DataFrame(index=tuneMultiIndex).reset_index()

    subsetTuneParameters = allTuneParameters.sample(n=10)

    subsetTuneParameters['AveragePerformance'] = [0] * len(subsetTuneParameters)
    subsetTuneParameters['TestsRun'] = [0] * len(subsetTuneParameters)

    kNearestOutput = pd.DataFrame()

    for currentParameter in subsetTuneParameters.index.tolist():

        print(subsetTuneParameters)

        for loopIndex in range(5):

            trainSet1Raw, trainSet2Raw = AuxML1.splitDataFrame(trainSetRaw, targets, .5, isReg)

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

            print("-----------------Starting the tune condensed Neighbor " + str(loopIndex) + " " + datetime.now().strftime("%d.%m.%Y_%I.%M.%S") + "-------------------------")
            print("***Condensing Test Set 1 " + datetime.now().strftime("%d.%m.%Y_%I.%M.%S") + "***")
            trainSet1Condensed = AuxML1.condensedNearestNeighbor(trainSet1, targets, 1, p, hybridCols, e, s, isReg)
            print("***Condensing Test Set 2 " + datetime.now().strftime("%d.%m.%Y_%I.%M.%S") + "***")
            trainSet2Condensed = AuxML1.condensedNearestNeighbor(trainSet2, targets, 1, p, hybridCols, e, s, isReg)

            kNearestTune = pd.DataFrame({
                'testID': [],
                'currentParameterID': [],
                'nearestNeighbors': [],
                'expectedValue': [],
                'actualValue': [],
                'correctAssignment': []
            })

            kNearestTune['expectedValue'] = kNearestTune['expectedValue'].astype('object')

            print("***Starting Tune " + str(subsetTuneParameters.loc[currentParameter, 'TestsRun']) + " for parameter set " + str(currentParameter) + "***")
            print("p = " + str(p))
            print("k = " + str(k))
            print("e = " + str(e))
            print("s = " + str(s))
            print("***" + datetime.now().strftime("%d.%m.%Y_%I.%M.%S") + "***")
            for currentTuneRow1 in tuneSet1.index.tolist():
                kNearestTune.loc[currentTuneRow1] = AuxML1.kNearestNeighbor(tuneSet1.loc[currentTuneRow1],
                                                                            trainSet1Condensed,
                                                                            targets,
                                                                            k,
                                                                            p,
                                                                            hybridCols,
                                                                            e,
                                                                            s,
                                                                            isReg)

            subsetTuneParameters.loc[currentParameter, 'TestsRun'] += 1
            subsetTuneParameters.to_csv(tuneFileName, index=True)

            kNearestTune['currentParameterID'] = currentParameter
            kNearestOutput = pd.concat([kNearestOutput, kNearestTune])
            kNearestOutput.to_csv(tuneRawFileName, index=True)

            # reset our table once tracked above before moving to the next index
            kNearestTune = kNearestTune.drop(kNearestTune.index.to_list())

            print("***Starting Tune " + str(subsetTuneParameters.loc[currentParameter, 'TestsRun']) + " for parameter set " + str(currentParameter) + "***")
            print("***" + datetime.now().strftime("%d.%m.%Y_%I.%M.%S") + "***")
            for currentTuneRow2 in tuneSet2.index.tolist():
                kNearestTune.loc[currentTuneRow2] = AuxML1.kNearestNeighbor(tuneSet2.loc[currentTuneRow2],
                                                                            trainSet2Condensed,
                                                                            targets,
                                                                            k,
                                                                            p,
                                                                            hybridCols,
                                                                            e,
                                                                            s,
                                                                            isReg)

            subsetTuneParameters.loc[currentParameter, 'TestsRun'] += 1
            subsetTuneParameters.to_csv(tuneFileName, index=True)

            kNearestTune['currentParameterID'] = currentParameter
            kNearestOutput = pd.concat([kNearestOutput, kNearestTune])
            kNearestOutput.to_csv(tuneRawFileName, index=True)

        subsetTuneParameters.loc[currentParameter, 'AveragePerformance'] = AuxML1.testEffectiveness(kNearestOutput[kNearestOutput['currentParameterID'] == currentParameter], isReg)
        subsetTuneParameters.to_csv(tuneFileName, index=True)

    subsetTuneParameters.to_csv(tuneFileName, index=True)
    trainSetRaw.to_csv(testCasesFileName, index=True)

    return subsetTuneParameters, trainSetRaw
