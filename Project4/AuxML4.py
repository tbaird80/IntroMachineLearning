import pandas as pd
import numpy as np
import TrackClass
from datetime import datetime
import random
import os

def createNewTestDirectory(trackType, learnType, discountFactor):
    uniqueTestDir = trackType + "Track/" + learnType + "/" + datetime.now().strftime("%d.%m.%Y_%I.%M.%S") + "_DF=" + str(discountFactor)
    if not os.path.exists(uniqueTestDir):
        # If the directory does not exist, create it
        os.makedirs(uniqueTestDir)
        print(f'Directory "{uniqueTestDir}" created.')
    else:
        print(f'Directory "{uniqueTestDir}" already exists.')

    return uniqueTestDir

def trainValueIteration(track):
    # create new directory for our training
    outputDirectory = createNewTestDirectory(track.trackType, track.learnType)
    stateOutput = outputDirectory + "/stateTable.csv"
    actionOutput = outputDirectory + "/actionTable.csv"

    print("***********************Updating Q Values " + datetime.now().strftime("%d.%m.%Y_%I.%M.%S") + "*****************************")
    track.updateQValuesVI()
    print("***********************Updating Value Table " + datetime.now().strftime("%d.%m.%Y_%I.%M.%S") + "*****************************")
    track.updateValueTable()

    # write table to memory for easy access next time
    track.stateTable.to_csv(stateOutput, index=True)
    track.actionTable.to_csv(actionOutput, index=True)

def trainQLearningSARSASubTrack(track, trainType='Q'):
    # create new directory for our training
    outputDirectory = createNewTestDirectory(track.trackType, track.learnType, track.discountFactor)
    stateOutput = outputDirectory + "/stateTable.csv"
    actionOutput = outputDirectory + "/actionTable.csv"
    historicalOutput = outputDirectory + "/historicalOutput.csv"

    # add in random Q values off the bat, store them down for reference later
    track.actionTable.loc[track.actionTable['QValue'].isna(), 'QValue'] = np.random.uniform(-.01, 0, track.actionTable['QValue'].isna().sum())
    track.updateValueTable()

    # init a few columns to help with tracking attempts
    if 'timesVisited' in track.actionTable.columns:
        track.actionTable.loc[:, 'timesVisited'] = np.where(track.actionTable['timesVisited'].isna(), 0, track.actionTable['timesVisited'])
        track.actionTable.loc[:, 'learningRate'] = np.where(track.actionTable['learningRate'].isna(), 1, track.actionTable['learningRate'])
    else:
        track.actionTable.loc[:, 'timesVisited'] = 0
        track.actionTable.loc[:, 'learningRate'] = 1

    for outerIndex in range(1001):
        # we are going to run it 10 times before updating value table
        for innerIndex in range(11):
            # print the map for reference
            # track.printCurrentMap()
            # find our starting action state as random state
            currentActionIndex = findStartingActionState(track, testType='TrainRandom')

            # our starting action state is S for starting line
            notFinished = True

            # run until we reach finish line
            while notFinished:
                #findCurrentState(track.actionTable.loc[currentActionIndex])

                # update our times visited and learning rate
                track.actionTable.loc[currentActionIndex, 'learningRate'] = 1 / (1 + track.actionTable.loc[currentActionIndex, 'timesVisited'])
                track.actionTable.loc[currentActionIndex, 'timesVisited'] += 1

                # find the next state given a successful action
                nextStateIndex = track.actionTable.loc[currentActionIndex, 'successValueMap']
                nextLandingState = track.stateTable.loc[nextStateIndex, 'locTypeState']

                # if we hit a finish line, there is no next action to consider
                if nextLandingState == 'F':
                    # grab prev Q Value and update it to new value accounting for next action
                    prevQValue = track.actionTable.loc[currentActionIndex, 'QValue']
                    track.actionTable.loc[currentActionIndex, 'QValue'] = prevQValue + track.actionTable.loc[currentActionIndex, 'learningRate'] * (-1 - prevQValue)
                    # we reached finish line, so the testing is done
                    notFinished = False

                # otherwise, we want to ensure that we take the next action into account when updating
                else:
                    # we need the best next action for Q calculation no matter what
                    nextStateActionBestIndex = findNextActionIndex(track, nextStateIndex, nextStepType='Success')

                    # either take the best or random depending on our epsilon search
                    epsilonGreedy = random.uniform(0, 1)
                    if epsilonGreedy > track.epsilon:
                        nextStateActionActualIndex = nextStateActionBestIndex
                    else:
                        nextStateActionActualIndex = findNextActionIndex(track, nextStateIndex, nextStepType='Random')

                    # grab our previous Q value (before update)
                    prevQValue = track.actionTable.loc[currentActionIndex, 'QValue']

                    # our next action Q value will be best option for Q learning and actual for SARSA
                    if trainType == 'Q':
                        nextActionQValue = track.actionTable.loc[nextStateActionBestIndex, 'QValue']
                    else:
                        nextActionQValue = track.actionTable.loc[nextStateActionActualIndex, 'QValue']

                    track.actionTable.loc[currentActionIndex, 'QValue'] = prevQValue + track.actionTable.loc[currentActionIndex, 'learningRate'] * (-1 + (track.discountFactor * nextActionQValue) - prevQValue)

                    currentActionIndex = nextStateActionActualIndex

        print("***********************Updating Value Table for " + track.trackFamily + str(track.smallerTrackID) + " **Round " + str(outerIndex) + " " + str(track.discountFactor) +
              "**" + datetime.now().strftime("%d.%m.%Y_%I.%M.%S") + "*****************************")
        track.updateValueTable()
        print(track.checkConvergence())
        # write table to memory for easy access next time
        track.stateTable.to_csv(stateOutput, index=True)
        track.actionTable.to_csv(actionOutput, index=True)
        track.historicalValues.to_csv(historicalOutput, index=True)

    finalOutputDirectory = track.trackType + "Track/" + track.learnType
    finalStateOutput = finalOutputDirectory + "/stateTable.csv"
    finalActionOutput = finalOutputDirectory + "/actionTable.csv"
    finalHistoricalOutput = finalOutputDirectory + "/historicalOutput.csv"

    track.stateTable.to_csv(finalStateOutput, index=True)
    track.actionTable.to_csv(finalActionOutput, index=True)
    track.historicalValues.to_csv(finalHistoricalOutput, index=True)

def readTrackFile(trackType):
    fileName = trackType + "-track.txt"

    storedTrack = []

    with open(fileName, 'r') as trackFile:
        # Read and discard the first line
        trackFile.readline()

        for currentLine in trackFile:
            # Strip the newline character at the end of each line
            strippedLine = currentLine.strip()
            # Split the line into individual characters and append to the 2D array
            storedTrack.append(list(strippedLine))

        xValues = []
        yValues = []
        currentValue = []

        for rowIndex in range(len(storedTrack)):
            for colIndex in range(len(storedTrack[0])):
                xValues.append(colIndex)
                yValues.append(rowIndex)
                currentValue.append(storedTrack[rowIndex][colIndex])

        coordinateOptions = pd.DataFrame({'xLoc': xValues,
                                          'yLoc': yValues,
                                          'locType': currentValue})

    return storedTrack, coordinateOptions

def bresenhamsAlgorithm(x0, y0, x1, y1):
    # Initialize the direction of the line
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -1 * abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    error = dx + dy

    locationList = []

    while True:
        locationList.append([x0, y0])
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * error
        if e2 >= dy:
            if x0 == x1:
                break
            error = error + dy
            x0 = x0 + sx
        if e2 <= dx:
            if y0 == y1:
                break
            error = error + dx
            y0 = y0 + sy

    return locationList

def nextTrackLoc(track, x0, y0, x1, y1, returnStart=False):
    # find the list of next locations
    locationList = bresenhamsAlgorithm(x0, y0, x1, y1)

    # init our output values
    returnX = 0
    returnY = 0
    movementType = '.'

    # iterate through all possible options of landing spots
    for currentLocIndex in range(len(locationList)):
        returnX = locationList[currentLocIndex][0]
        returnY = locationList[currentLocIndex][1]
        nextTrackLocation = track[(track['xLoc'] == returnX) & (track['yLoc'] == returnY)]
        movementType = nextTrackLocation['locType'].values[0]

        # if we hit final spot, this is end of road
        if movementType == 'F':
            return returnX, returnY, movementType
        # if we hit wall, adjust one location index before
        elif movementType == '#':
            if returnStart:
                returnX, returnY = findClosestStart(track, x0, y0)
            else:
                returnX = locationList[currentLocIndex - 1][0]
                returnY = locationList[currentLocIndex - 1][1]
            return returnX, returnY, movementType

    return [returnX, returnY, movementType]

def findClosestStart(track, x0, y0):
    # find the location start coordinates, add to list
    startStateLocations = track[track['locType'] == 'S']
    startLocations = []
    for currentLocation in startStateLocations.index.tolist():
        xValue = startStateLocations.loc[currentLocation, 'xLoc']
        yValue = startStateLocations.loc[currentLocation, 'yLoc']
        startLocations.append([xValue, yValue])

    # find the smallest list size to represent the closest start value to where we left track
    smallestListSize = 1000
    for currentStartValues in startLocations:
        listSize = len(bresenhamsAlgorithm(x0, y0, currentStartValues[0], currentStartValues[1]))
        if listSize < smallestListSize:
            smallestListSize = listSize
            returnX = currentStartValues[0]
            returnY = currentStartValues[1]

    return returnX, returnY

def findNextActionIndex(track, nextActionIndex, nextStepType='Success'):
    # filter our action table by the current state in question
    currentStateActions = track.actionTable[track.actionTable['currentStateValueMap'] == nextActionIndex]

    # if we had a successful action, then we want the best action choice of that state
    if nextStepType == 'Success':
        nextStateActionIndex = currentStateActions['QValue'].idxmax()
    else:
        nextStateActionIndex = currentStateActions.sample(n=1).index[0]

    return nextStateActionIndex

def findStartingActionState(track, testType='Test'):
    # find only our next starting spot
    if testType == 'TrainRandom':
        startingOptions = track.actionTable[(track.actionTable['landingTypeCurrent'] == 'S')]
        firstStateIndex = startingOptions.sample(n=1).index[0]
    elif testType == 'TrainBest':
        startingOptions = track.actionTable[(track.actionTable['landingTypeCurrent'] == 'S')]
        firstStateIndex = startingOptions['QValue'].idxmax()
    else:
        startingOptions = track.actionTable[(track.actionTable['landingTypeCurrent'] == 'S') & (track.stateTable['xVel'] == 0) & (track.stateTable['yVel'] == 0)]
        firstStateIndex = startingOptions['QValue'].idxmax()

    return firstStateIndex

def findCurrentState(currentActionRecord):
    currentX = currentActionRecord['xLoc']
    currentY = currentActionRecord['yLoc']
    currentXVel = currentActionRecord['xVel']
    currentYVel = currentActionRecord['yVel']
    currentXAccel = currentActionRecord['xAccel']
    currentYAccel = currentActionRecord['yAccel']

    print(f"We have a current coordinate of [{currentX}, {currentY}], speed of [{currentXVel}, {currentYVel}], and acceleration of [{currentXAccel}, {currentYAccel}]")
