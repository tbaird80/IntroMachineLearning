import pandas as pd
import numpy as np
import TrackClass
from datetime import datetime
import random
import os

def createNewTestDirectory(trackType, learnType, discountFactor):
    """
    Create the new test directory

    @param trackType: the current track type
    @param learnType: the type of learning that we are employing
    @param discountFactor: the current discount factor
    @return: return the new folder path
    """

    # logic to help name the track test files
    if len(trackType) == 1:
        uniqueTestDir = trackType + "Track/" + learnType + "/" + datetime.now().strftime("%d.%m.%Y_%I.%M.%S") + "_DF=" + str(discountFactor)
    else:
        uniqueTestDir = trackType + "Track/" + learnType + "/" + datetime.now().strftime("%d.%m.%Y_%I.%M.%S") + "_DF=" + str(discountFactor)

    # create file if it doesnt exist
    if not os.path.exists(uniqueTestDir):
        # If the directory does not exist, create it
        os.makedirs(uniqueTestDir)
        print(f'Directory "{uniqueTestDir}" created.')
    else:
        print(f'Directory "{uniqueTestDir}" already exists.')

    return uniqueTestDir

def trainValueIteration(track, currentTestID):
    """
    Train a value iteration track

    @param track: the current track object
    @param currentTestID: the current test that we are running
    @return: nothing to return
    """

    # create new directory for our training
    outputDirectory = createNewTestDirectory(track.trackType, track.learnType, track.discountFactor)
    stateOutput = outputDirectory + "/stateTable.csv"
    actionOutput = outputDirectory + "/actionTable.csv"

    print(f"***********************Updating Q Values **{currentTestID}** " + datetime.now().strftime("%d.%m.%Y_%I.%M.%S") + "*****************************")
    track.updateQValuesVI()
    print(f"***********************Updating Value Table **{currentTestID}** " + datetime.now().strftime("%d.%m.%Y_%I.%M.%S") + "*****************************")
    track.updateValueTable()

    # write table to memory for easy access next time
    track.stateTable.to_csv(stateOutput, index=True)
    track.actionTable.to_csv(actionOutput, index=True)

def trainQLearningSARSASubTrack(track, trainType='QLearning'):
    """
    Wrapper function train a Q learning or SARSA track

    @param track: the current track type
    @param trainType: the type of training
    @return: returns nothing
    """

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

    numberOfTests = 101

    for outerIndex in range(numberOfTests):
        # we are going to run it 10 times before updating value table
        for innerIndex in range(11):
            # print the map for reference
            track.printCurrentMap()
            # find our starting action state as random state
            if track.smallerTrackID == 0:
                currentActionIndex = findStartingActionState(track, testType='Best')
            else:
                currentActionIndex = findStartingActionState(track, testType='TrainRandom')

            # our starting action state is S for starting line
            notFinished = True
            currentActionLoopIndex = 0

            # run until we reach finish line
            while notFinished:
                findCurrentState(track, currentActionIndex)

                # update our times visited and learning rate
                track.actionTable.loc[currentActionIndex, 'learningRate'] = (1 * track.tau) / (track.tau + track.actionTable.loc[currentActionIndex, 'timesVisited'])
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
                    if epsilonGreedy > .2:
                        nextStateActionActualIndex = nextStateActionBestIndex
                    else:
                        nextStateActionActualIndex = findNextActionIndex(track, nextStateIndex, nextStepType='Random')

                    # grab our previous Q value (before update)
                    prevQValue = track.actionTable.loc[currentActionIndex, 'QValue']

                    # our next action Q value will be best option for Q learning and actual for SARSA
                    if trainType == 'QLearning':
                        nextActionQValue = track.actionTable.loc[nextStateActionBestIndex, 'QValue']
                    else:
                        nextActionQValue = track.actionTable.loc[nextStateActionActualIndex, 'QValue']

                    # update Q value
                    track.actionTable.loc[currentActionIndex, 'QValue'] = prevQValue + track.actionTable.loc[currentActionIndex, 'learningRate'] * (-1 + (track.discountFactor * nextActionQValue) - prevQValue)

                    # update our current action to next action
                    currentActionIndex = nextStateActionActualIndex

                    # iterate our loop
                    currentActionLoopIndex += 1

                    # stop running if we reach max iteration of 1000
                    if currentActionLoopIndex > 1000:
                        notFinished = False

        print("***********************Updating Value Table for " + track.trackFamily + " " + track.learnType + " " + str(track.tau) + " " + str(track.discountFactor)
              + " **Smaller Track " + str(track.smallerTrackID) + ": Round " + str(outerIndex) +
              "**" + datetime.now().strftime("%d.%m.%Y_%I.%M.%S") + "*****************************")
        track.updateValueTable()
        print(track.checkConvergence())
        # write table to memory for easy access next time
        track.stateTable.to_csv(stateOutput, index=True)
        track.actionTable.to_csv(actionOutput, index=True)
        track.historicalValues.to_csv(historicalOutput, index=True)

    # write to the current test set
    finalOutputDirectory = track.trackType + "Track/" + track.learnType
    finalStateOutput = finalOutputDirectory + "/stateTable.csv"
    finalActionOutput = finalOutputDirectory + "/actionTable.csv"
    finalHistoricalOutput = finalOutputDirectory + "/historicalOutput.csv"

    track.stateTable.to_csv(finalStateOutput, index=True)
    track.actionTable.to_csv(finalActionOutput, index=True)
    track.historicalValues.to_csv(finalHistoricalOutput, index=True)

    # write to the overall test set
    finalOutputDirectory = track.trackFamily + "Track/" + track.learnType
    finalStateOutput = finalOutputDirectory + "/stateTable.csv"
    finalActionOutput = finalOutputDirectory + "/actionTable.csv"
    finalHistoricalOutput = finalOutputDirectory + "/historicalOutput.csv"

    track.stateTable.to_csv(finalStateOutput, index=True)
    track.actionTable.to_csv(finalActionOutput, index=True)
    track.historicalValues.to_csv(finalHistoricalOutput, index=True)

def readTrackFile(trackType):
    """
    Find our desired track table

    @param trackType: the track type
    @return: the stored track and coordinate options
    """

    # find the desired file
    fileName = trackType + "-track.txt"

    # init our track
    storedTrack = []

    # read every line of the track
    with open(fileName, 'r') as trackFile:
        # Read and discard the first line since it only has dimensions
        trackFile.readline()

        for currentLine in trackFile:
            # Strip the newline character at the end of each line
            strippedLine = currentLine.strip()
            # Split the line into individual characters and append to the 2D array
            storedTrack.append(list(strippedLine))

        # init our X and Y values
        xValues = []
        yValues = []
        currentValue = []

        # add all of our values to lists
        for rowIndex in range(len(storedTrack)):
            for colIndex in range(len(storedTrack[0])):
                xValues.append(colIndex)
                yValues.append(rowIndex)
                currentValue.append(storedTrack[rowIndex][colIndex])

        # create our table with coordinate options
        coordinateOptions = pd.DataFrame({'xLoc': xValues,
                                          'yLoc': yValues,
                                          'locType': currentValue})

    return storedTrack, coordinateOptions

def bresenhamsAlgorithm(x0, y0, x1, y1):
    """
    The algorithm that helps us find all of the landing spots between two proposed locations. This was adapted from the provided pseudocode
    from this link:
    https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm

    @param x0: starting x
    @param y0: starting y
    @param x1: finish x
    @param y1: finish y
    @return: the list of landing spots
    """

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
    """
    Used in creating our action table, allows us to find the actual final landing spot accounting for walls.


    @param track: the current track object
    @param x0: the current x
    @param y0: the current y
    @param x1: the proposed x
    @param y1: the proposed y
    @param returnStart: whether or not we need to return to start on hitting a wall
    @return: the X, Y, and location type
    """

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
    """
    On crash track, we need to return to closest starting spot

    @param track: the current track object
    @param x0: the current x location
    @param y0: the current y location
    @return: the closest starting index location
    """

    # find the location start coordinates, add to list
    startStateLocations = track[track['locType'] == 'S']
    startLocations = []

    # find all of the start locations
    for currentLocation in startStateLocations.index.tolist():
        xValue = startStateLocations.loc[currentLocation, 'xLoc']
        yValue = startStateLocations.loc[currentLocation, 'yLoc']
        startLocations.append([xValue, yValue])

    # find the smallest list size to represent the closest start value to where we left track
    smallestListSize = 1000

    # iterate through each start spot and find the closest
    for currentStartValues in startLocations:
        listSize = len(bresenhamsAlgorithm(x0, y0, currentStartValues[0], currentStartValues[1]))
        if listSize < smallestListSize:
            smallestListSize = listSize
            returnX = currentStartValues[0]
            returnY = currentStartValues[1]

    return returnX, returnY

def findNextActionIndex(track, nextStateIndex, nextStepType='Success'):
    """
    find the next action to take

    @param track: the current track object
    @param nextStateIndex: the current state that we are in
    @param nextStepType: the next step type that we are going for
    @return: the next state action pair
    """

    # filter our action table by the current state in question
    currentStateActions = track.actionTable[track.actionTable['currentStateValueMap'] == nextStateIndex]

    # if we had a successful action, then we want the best action choice of that state
    if nextStepType == 'Success':
        nextStateActionIndex = currentStateActions['QValue'].idxmax()
        # check for loop, set to random if so
        if nextStateIndex == track.actionTable.loc[nextStateActionIndex, 'successValueMap'] == track.actionTable.loc[nextStateActionIndex, 'failValueMap']:
            nextStateActionIndex = currentStateActions.sample(n=1).index[0]
    # pick a random action
    else:
        nextStateActionIndex = currentStateActions.sample(n=1).index[0]

    return nextStateActionIndex

def findStartingActionState(track, testType='Best'):
    """
    Find a good starting action state

    @param track: the current state action pair
    @param testType: what type of test we are running
    @return: the index of the desired state action pair
    """

    # find only our next starting spot
    # truly random start, any velocity
    if testType == 'TrainRandom':
        startingOptions = track.actionTable[(track.actionTable['landingTypeCurrent'] == 'S')]
        firstStateIndex = startingOptions.sample(n=1).index[0]
    # random start but velocity of 0
    elif testType == 'TrainRandomActualStart':
        startingOptions = track.actionTable[(track.actionTable['landingTypeCurrent'] == 'S') & (track.actionTable['xVel'] == 0) & (track.actionTable['yVel'] == 0)]
        firstStateIndex = startingOptions.sample(n=1).index[0]
    # best possible start but any velocity
    elif testType == 'TrainBest':
        startingOptions = track.actionTable[(track.actionTable['landingTypeCurrent'] == 'S')]
        firstStateIndex = startingOptions['QValue'].idxmax()
    # best possible start but 0 velocity
    elif testType == 'TrainBestActualStart':
        startingOptions = track.actionTable[(track.actionTable['landingTypeCurrent'] == 'S') & (track.actionTable['xVel'] == 0) & (track.actionTable['yVel'] == 0)]
        firstStateIndex = startingOptions['QValue'].idxmax()

    return firstStateIndex

def findCurrentState(currentTrack, currentActionIndex):
    """
    Visualization function to show the curent state of our actor

    @param currentTrack: the current track object
    @param currentActionIndex: the current action state pair
    @return: no return
    """

    # find all the current action state characteristics
    currentActionRecord = currentTrack.actionTable.loc[currentActionIndex]
    currentX = currentActionRecord['xLoc']
    currentY = currentActionRecord['yLoc']
    currentXVel = currentActionRecord['xVel']
    currentYVel = currentActionRecord['yVel']
    currentXAccel = currentActionRecord['xAccel']
    currentYAccel = currentActionRecord['yAccel']
    currentState = currentActionRecord['currentStateValueMap']
    nextStateSuccess = currentActionRecord['successValueMap']
    nextStateFail = currentActionRecord['failValueMap']

    # print the characteristics
    print(f"At {currentActionIndex}, we have a current coordinate of [{currentX}, {currentY}], speed of [{currentXVel}, {currentYVel}], and acceleration of [{currentXAccel}, {currentYAccel}]")
    print(f"Currently at {currentState}, landing spots are Success: {nextStateSuccess} and Fail: {nextStateFail}")

    # print the track
    currentTrack.printCurrentMap(currentX, currentY)
