import pandas as pd
from datetime import datetime
import AuxML4 as Aux
import random
import os

def runTests(track, trainType='QLearning'):
    # run first test
    lastTestOutput = testFullTrack(track)

    # create our output file for our tests
    testOutputFilePath = track.trackType + "Track/" + track.learnType + "/testOutputsARCHIVE.csv"
    fullTestTable = findTestTable(testOutputFilePath)

    # train it 25 more times for good measure
    for index in range(1):
        print("***********************Next Final Test Set for " + trainType + " " + track.trackFamily + "**Round " + str(index) +
              "**" + datetime.now().strftime("%d.%m.%Y_%I.%M.%S") + "*****************************")
        fullTestTable = writeTestTable(track, testOutputFilePath, fullTestTable, lastTestOutput)
        # Aux.trainQLearningSARSASubTrack(track, trainType=trainType)
        # lastTestOutput = testFullTrack(track)

def findTestTable(testFilePath):
    if os.path.exists(testFilePath):
        print("Test file exists, reading in previous tests")
        testFileTable = pd.read_csv(testFilePath, index_col=0)
    else:
        print("Test does not exists, creating new table")
        testFileTable = pd.DataFrame()

    return testFileTable

def writeTestTable(track, testFilePath, testFile, newTest):
    # Initialize an empty dictionary to hold the new row data
    newRow = {}

    newRow['tau'] = track.tau
    newRow['DF'] = track.discountFactor

    # Loop through each test case
    for x in range(10):
        # Define the column name
        columnName = 'Test' + str(x + 1)
        newRow[columnName] = newTest[x]

    # Calculate the statistics
    min_val = min(newTest)
    max_val = max(newTest)
    mean_val = sum(newTest) / len(newTest)

    # Calculate mean excluding min and max
    mean_excl_min_max = (sum(newTest) - min_val - max_val) / (len(newTest) - 2)

    # Add the statistics to the newRow dictionary
    newRow['Min'] = min_val
    newRow['Max'] = max_val
    newRow['Mean'] = mean_val
    newRow['AdjMean'] = mean_excl_min_max

    # Convert the dictionary to a DataFrame and append it to the testResults DataFrame
    testFile = pd.concat([testFile, pd.DataFrame([newRow])])

    testFile.to_csv(testFilePath, index=True)

    return testFile

def testFullTrack(track):
    # create new directory for our training
    testCasesOutput = []

    print("**Starting Test Cases**")

    for testIndex in range(11):
        # find our starting action state as random state
        currentActionIndex = Aux.findStartingActionState(track, testType='Best')
        stepCounter = 0

        # our starting action state is S for starting line
        notFinished = True

        # run until we reach finish line
        while notFinished:
            # Aux.findCurrentState(track, currentActionIndex)
            stepCounter += 1

            nextStateIndex = moveSpace(track, currentActionIndex)
            currentLandingState = track.stateTable.loc[nextStateIndex, 'locTypeState']
            if currentLandingState == 'F' or stepCounter > 100:
                notFinished = False
            else:
                currentActionIndex = Aux.findNextActionIndex(track, nextStateIndex, nextStepType='Success')

        print("Complete Track " + str(testIndex) + " in " + str(stepCounter) + " steps")
        testCasesOutput.append(stepCounter)

    return testCasesOutput

def moveSpace(track, currentActionStateIndex):
    failAccelTest = random.uniform(0, 1)

    if failAccelTest > .2:
        nextLocationIndex = track.actionTable.loc[currentActionStateIndex, 'successValueMap']
    else:
        nextLocationIndex = track.actionTable.loc[currentActionStateIndex, 'failValueMap']

    return nextLocationIndex
