import pandas as pd
from datetime import datetime
import AuxML4 as Aux
import random
import os

def runTests(track, trainType='QLearning'):
    """
    A wrapper function that allows us to perform tests on q learning and sarsa types

    @param track: the current track object
    @param trainType: the training type that we are employing
    @return: no return
    """

    # run first test
    lastTestOutput = testFullTrack(track, batchIndex=0)

    # init our previous test mean
    lastMean = 0

    # create our output file for our tests
    testOutputFilePath = track.trackType + "Track/" + track.learnType + "/testOutputsFINAL.csv"
    fullTestTable = findTestTable(testOutputFilePath)

    # train it 10 more times for good measure
    for index in range(10):
        # write most recent result to test table
        fullTestTable = writeTestTable(track, testOutputFilePath, fullTestTable, lastTestOutput)

        # check for mean difference as a way to test for convergence
        currentMean = sum(lastTestOutput) / len(lastTestOutput)
        meanDiff = abs(lastMean - currentMean)

        # print the update
        print()
        print("The mean of the current test set is: " + str(currentMean))
        print("The current change in performance is: " + str(meanDiff))
        print()

        # check for convergence, keep going if not
        if meanDiff > 3:
            # reset our mean values
            lastMean = currentMean
            Aux.trainQLearningSARSASubTrack(track, trainType=trainType)
            lastTestOutput = testFullTrack(track, batchIndex=index)
        # break if so
        else:
            break

    return True

def runTestsVI(track):
    """
    A wrapper function to run tests on our value iteration test sets

    @param track: the current track in question
    @return: no return
    """

    # create our output file for our tests
    testOutputFilePath = track.trackType + "Track/" + track.learnType + "/testOutputsFINAL" + str(track.discountFactor) + ".csv"
    valueOutputFilePath = track.trackType + "Track/" + track.learnType + "/valueTableFINAL" + str(track.discountFactor) + ".csv"
    fullTestTable = findTestTable(testOutputFilePath)

    # allow for upwards of 1000 tests
    for index in range(1000):
        # train another round of tests and check for convergence
        Aux.trainValueIteration(track, index + 1)
        changeInValue = track.checkConvergence()

        print(f"Our current change in value is: {changeInValue}")

        # if our change is less than a small value, we assume convergence and end our testing
        if changeInValue < .001:
            break

    # test trained track
    lastTestOutput = testFullTrack(track, batchIndex=0)

    # write most test result to test table and
    writeTestTable(track, testOutputFilePath, fullTestTable, lastTestOutput, index+1)
    track.historicalValues.to_csv(valueOutputFilePath, index=True)

    return True

def findTestTable(testFilePath):
    """
    Create file path for test storage if not present. If present, return the existing table

    @param testFilePath: file path that we are looking to create
    @return: return the created file path
    """

    # check if file exists, read file if it does
    if os.path.exists(testFilePath):
        print("Test file exists, reading in previous tests")
        testFileTable = pd.read_csv(testFilePath, index_col=0)
    # otherwise create new table
    else:
        print("Test does not exists, creating new table")
        testFileTable = pd.DataFrame()

    return testFileTable

def writeTestTable(track, testFilePath, testFile, newTest, batchTestID):
    """
    Helper function that helps us store our tests

    @param track: the current track object
    @param testFilePath: the file path that we want to store our results on
    @param testFile: the existing test results
    @param newTest: the most recent tests run
    @param batchTestID: the current batch of tests
    @return: the updated test table
    """

    # Initialize an empty dictionary to hold the new row data
    newRow = {}

    # add columns that represent the test case
    newRow['tau'] = track.tau
    newRow['DF'] = track.discountFactor
    newRow['testRun'] = batchTestID

    # Loop through each test case
    for x in range(100):
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

    # write to file
    testFile.to_csv(testFilePath, index=True)

    return testFile

def testFullTrack(track, batchIndex):
    """
    Allow us to run a test on the trained track

    @param track: the current track object
    @param batchIndex: the current batch of tests
    @return: the updated test table
    """

    print("***********************Next Final Test Set for " + track.learnType + " " + track.trackFamily + "**Round " + str(batchIndex) +
          "**" + datetime.now().strftime("%d.%m.%Y_%I.%M.%S") + "*****************************")

    # create new directory for our training
    testCasesOutput = []

    print("**Starting Test Cases**")

    # run 100 tests till completion
    for testIndex in range(101):
        # find our starting action state as random state
        currentActionIndex = Aux.findStartingActionState(track, testType='Best')
        stepCounter = 0

        # our starting action state is S for starting line
        notFinished = True

        # run until we reach finish line
        while notFinished:
            # print the current state for visualization purposes and increment our tests
            Aux.findCurrentState(track, currentActionIndex)
            stepCounter += 1

            # find the next space and check if it is a finish spot
            nextStateIndex = moveSpace(track, currentActionIndex)
            currentLandingState = track.stateTable.loc[nextStateIndex, 'locTypeState']

            # if finish or we have reached a max number of tests, end the test run
            if currentLandingState == 'F' or stepCounter > 200:
                notFinished = False
            else:
                currentActionIndex = Aux.findNextActionIndex(track, nextStateIndex, nextStepType='Success')

        print("Complete Track " + str(testIndex) + " in " + str(stepCounter) + " steps")
        testCasesOutput.append(stepCounter)

    return testCasesOutput

def moveSpace(track, currentActionStateIndex):
    """
    Move our actor to the next space. Incorporates a 20% chance that our intended action fails

    @param track: the current track object
    @param currentActionStateIndex: the current state action pair
    @return: the next state location
    """

    # find random number
    failAccelTest = random.uniform(0, 1)

    # 80% of the time take the successful action state
    if failAccelTest > .2:
        nextLocationIndex = track.actionTable.loc[currentActionStateIndex, 'successValueMap']
        print("\nOur acceleration SUCCEEDS\n")
    # 20% of the time, take the failed action state
    else:
        nextLocationIndex = track.actionTable.loc[currentActionStateIndex, 'failValueMap']
        print("\nOur acceleration FAILS\n")

    return nextLocationIndex
