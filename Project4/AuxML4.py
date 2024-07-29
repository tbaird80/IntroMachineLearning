import pandas as pd
import numpy as np
import TrackClass

def runValueIteration:
    # create

def bresenhamsAlgorithm(x0, y0, x1, y1):
    # Initialize the direction of the line
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -1 * abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    error = dx + dy

    returnList = []

    while True:
        returnList.append([x0, y0])
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

    return returnList


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

    return coordinateOptions


def nextTrackLoc(track, nextLocations):
    returnX = 0
    returnY = 0
    movementType = '.'

    for currentLocIndex in range(len(nextLocations)):
        returnX = nextLocations[currentLocIndex][0]
        returnY = nextLocations[currentLocIndex][1]
        nextTrackLocation = track.currentTrack[(track.currentTrack['xLoc'] == returnX) & (track.currentTrack['yLoc'] == returnY)]
        movementType = nextTrackLocation['locType'].values[0]

        if movementType == 'F':
            return returnX, returnY, movementType
        elif movementType == '#':
            returnX = nextLocations[currentLocIndex - 1][0]
            returnY = nextLocations[currentLocIndex - 1][1]
            return returnX, returnY, movementType

    return [returnX, returnY, movementType]

def findNextActionIndex(track, currentActionIndex, nextStepType='Success'):
    # filter our action table by the current state in question
    currentStateActions = track.actionTable[track.stateTable['currentStateValueMap'] == currentActionIndex]

    # if we had a successful action, then we want the
    if nextStepType == 'Success':
        maxQAction = currentStateActions[['successValueMap', 'QValue']].groupby(['currentStateValueMap'], as_index=False).max('QValue')
        nextStateIndex = maxQAction['successValueMap'].values[0]
    elif nextStepType == 'Fail':
        maxQAction = currentStateActions[['failValueMap', 'QValue']].groupby(['currentStateValueMap'], as_index=False).max('QValue')
        nextStateIndex = maxQAction['failValueMap'].values[0]
    else:
        maxQAction = currentStateActions[['successValueMap', 'QValue']].sample(n=1)
        nextStateIndex = maxQAction['successValueMap'].values[0]

    return nextStateIndex

