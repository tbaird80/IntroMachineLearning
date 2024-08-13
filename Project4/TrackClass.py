import AuxML4 as Aux
import pandas as pd
import numpy as np
from datetime import datetime
import os


class Track:

    def __init__(self, trackName, trackFamily, learnType, learningRate=0, discountFactor=0, epsilon=.1, returnStart=False, smallerTrackID=0):
        rawTrack, trackTable = Aux.readTrackFile(trackName)
        self.rawTrack = rawTrack
        self.currentTrack = trackTable
        self.trackType = trackName
        self.trackFamily = trackFamily
        self.learnType = learnType
        self.learningRate = learningRate
        self.discountFactor = discountFactor
        self.epsilon = epsilon
        self.historicalValues = pd.DataFrame({'EpochNumber': [],
                                              'SumOfValues': []})
        self.returnStart = returnStart
        self.smallerTrackID = smallerTrackID

        print("***********************Starting to build State Table for " + trackName + " " + datetime.now().strftime("%d.%m.%Y_%I.%M.%S") + "*****************************")
        self.stateTable = self.createStateTable()

        print("***********************Starting to build Action Table for " + trackName + " " + datetime.now().strftime("%d.%m.%Y_%I.%M.%S") + "*****************************")
        self.actionTable = self.createActionTable()

        # update our tables to account for previous runs
        if self.smallerTrackID > 1:
            self.updateForPreviousRuns()

    def updateForPreviousRuns(self):
        # find the results of the previous run
        prevRun = self.smallerTrackID - 1
        savedDirectory = "Smaller" + str(prevRun) + self.trackFamily + "Track/" + self.learnType
        prevActionTablePath = savedDirectory + "/actionTable.csv"
        prevHistoricalTablePath = savedDirectory + "/historicalOutput.csv"
        prevStateTablePath = savedDirectory + "/stateTable.csv"

        # read from local directory
        prevActionTable = pd.read_csv(prevActionTablePath, index_col=0)
        prevHistoricalTable = pd.read_csv(prevHistoricalTablePath, index_col=0)
        prevStateTable = pd.read_csv(prevStateTablePath, index_col=0)

        self.actionTable = self.actionTable.drop(columns=['QValue'])
        self.actionTable = self.actionTable.merge(prevActionTable['xLoc', 'yLoc', 'xVel', 'yVel', 'xAccel', 'yAccel', 'QValue', 'timesVisited', 'learningRate'],
                                                  on=['xLoc', 'yLoc', 'xVel', 'yVel', 'xAccel', 'yAccel'])

        self.historicalValues = prevHistoricalTable

        self.stateTable = self.stateTable.drop(columns=['currentValue'])
        self.stateTable = self.stateTable.merge(prevStateTable['xLocState', 'yLocState', 'xVelState', 'yVelState', 'currentValue'],
                                                on=['xLocState', 'yLocState', 'xVelState', 'yVelState', ])

    def createStateTable(self):
        savedDirectory = self.trackType + "Track"
        stateTableFilePath = savedDirectory + "/StateTable.csv"

        if os.path.exists(stateTableFilePath):
            print(stateTableFilePath + " data exists, reading from csv")

            # read from local directory
            fullStateTable = pd.read_csv(stateTableFilePath, index_col=0)
            return fullStateTable
        elif not os.path.exists(savedDirectory):
            print("Creating directory:" + savedDirectory)
            os.makedirs(savedDirectory)

        # find the coordinates that are valid resting points
        validStateCoordinates = self.currentTrack[self.currentTrack.locType.isin(['S', '.', 'F'])]

        # all possible states to pair with coordinate options
        stateDict = {'xVel': list(range(-5, 6)), 'yVel': list(range(-5, 6))}

        # create all possible state options
        stateOptions = pd.MultiIndex.from_product(stateDict.values(), names=stateDict.keys())
        stateOptions = pd.DataFrame(index=stateOptions).reset_index()
        fullStateTable = validStateCoordinates.merge(stateOptions, how='cross')

        # rename columns as needed
        fullStateTable.columns = ['xLocState', 'yLocState', 'locTypeState', 'xVelState', 'yVelState']

        # create new columns for reference later
        fullStateTable.loc[:, 'indexMap'] = fullStateTable.index
        fullStateTable.loc[:, 'currentValue'] = 0

        # write table to memory for easy access next time
        fullStateTable.to_csv(stateTableFilePath, index=True)
        return fullStateTable

    def createActionTable(self):
        stateActionTableFilePath = self.trackType + "Track/StateActionTable.csv"

        # check if our table already exists
        if os.path.exists(stateActionTableFilePath):
            print(stateActionTableFilePath + " data exists, reading from csv")

            # read from local directory
            fullStateActionTable = pd.read_csv(stateActionTableFilePath, index_col=0)
            return fullStateActionTable

        # find the coordinate points that are valid action spaces
        validCoordinates = self.currentTrack[self.currentTrack.locType.isin(['S', '.'])]

        # create dictionary of all possible states and actions
        stateActionDict = {'xVel': list(range(-5, 6)), 'yVel': list(range(-5, 6)), 'xAccel': [-1, 0, 1], 'yAccel': [-1, 0, 1]}

        # create table of all possible state/action pairs
        stateActionOptions = pd.MultiIndex.from_product(stateActionDict.values(), names=stateActionDict.keys())
        stateActionOptions = pd.DataFrame(index=stateActionOptions).reset_index()

        # find the adjusted velocity to account for the proposed acceleration action in both the X and Y direction
        stateActionOptions.loc[:, 'xVelAdj'] = stateActionOptions['xVel'] + stateActionOptions['xAccel']
        stateActionOptions.loc[:, 'xVelAdj'] = np.where(stateActionOptions['xVelAdj'] < -5, -5, stateActionOptions['xVelAdj'])
        stateActionOptions.loc[:, 'xVelAdj'] = np.where(stateActionOptions['xVelAdj'] > 5, 5, stateActionOptions['xVelAdj'])
        stateActionOptions.loc[:, 'yVelAdj'] = stateActionOptions['yVel'] + stateActionOptions['yAccel']
        stateActionOptions.loc[:, 'yVelAdj'] = np.where(stateActionOptions['yVelAdj'] < -5, -5, stateActionOptions['yVelAdj'])
        stateActionOptions.loc[:, 'yVelAdj'] = np.where(stateActionOptions['yVelAdj'] > 5, 5, stateActionOptions['yVelAdj'])

        # merge in coordinates to give full table of actions/coordinates
        fullStateActionTable = validCoordinates.merge(stateActionOptions, how='cross')

        # subset table as there are overlaps in state/action/result
        condensedStateActionTable = fullStateActionTable[['xLoc', 'yLoc', 'xVel', 'yVel', 'xVelAdj', 'yVelAdj']].drop_duplicates()

        # find the next possible location based on both a successful acceleration and not successful acceleration
        condensedStateActionTable.loc[:, 'xLocNextSuccess'] = condensedStateActionTable['xLoc'] + condensedStateActionTable['xVelAdj']
        condensedStateActionTable.loc[:, 'yLocNextSuccess'] = condensedStateActionTable['yLoc'] + condensedStateActionTable['yVelAdj']
        condensedStateActionTable.loc[:, 'xLocNextFail'] = condensedStateActionTable['xLoc'] + condensedStateActionTable['xVel']
        condensedStateActionTable.loc[:, 'yLocNextFail'] = condensedStateActionTable['yLoc'] + condensedStateActionTable['yVel']

        # find all the intermediate locations between proposed start and end locations on map
        print("**Bresenham Algo for Success " + self.trackType + " " + datetime.now().strftime("%d.%m.%Y_%I.%M.%S") + "**")
        condensedStateActionTable.loc[:, 'landingSpotSuccess'] = condensedStateActionTable.apply(lambda row: Aux.nextTrackLoc(self.currentTrack,
                                                                                                                              row['xLoc'],
                                                                                                                              row['yLoc'],
                                                                                                                              row['xLocNextSuccess'],
                                                                                                                              row['yLocNextSuccess'],
                                                                                                                              self.returnStart), axis=1)

        print("**Bresenham Algo for Fail " + self.trackType + " " + datetime.now().strftime("%d.%m.%Y_%I.%M.%S") + "**")
        condensedStateActionTable.loc[:, 'landingSpotFail'] = condensedStateActionTable.apply(lambda row: Aux.nextTrackLoc(self.currentTrack,
                                                                                                                           row['xLoc'],
                                                                                                                           row['yLoc'],
                                                                                                                           row['xLocNextFail'],
                                                                                                                           row['yLocNextFail'],
                                                                                                                           self.returnStart), axis=1)

        print("**Finding the relevant landing descriptions " + self.trackType + " " + datetime.now().strftime("%d.%m.%Y_%I.%M.%S") + "**")

        # pull out the relevant X/Y/type of the actual landing spot calculated above
        condensedStateActionTable.loc[:, 'nextXSuccess'] = condensedStateActionTable.apply(lambda row: row['landingSpotSuccess'][0], axis=1)
        condensedStateActionTable.loc[:, 'nextYSuccess'] = condensedStateActionTable.apply(lambda row: row['landingSpotSuccess'][1], axis=1)
        condensedStateActionTable.loc[:, 'landingTypeSuccess'] = condensedStateActionTable.apply(lambda row: row['landingSpotSuccess'][2], axis=1)
        condensedStateActionTable.loc[:, 'nextXFail'] = condensedStateActionTable.apply(lambda row: row['landingSpotFail'][0], axis=1)
        condensedStateActionTable.loc[:, 'nextYFail'] = condensedStateActionTable.apply(lambda row: row['landingSpotFail'][1], axis=1)
        condensedStateActionTable.loc[:, 'landingTypeFail'] = condensedStateActionTable.apply(lambda row: row['landingSpotFail'][2], axis=1)

        # adjust the resulting velocity measure to reset to 0, 0 when it hits a wall
        condensedStateActionTable.loc[:, 'nextXVelSuccess'] = np.where(condensedStateActionTable['landingTypeSuccess'] == '#', 0, condensedStateActionTable['xVelAdj'])
        condensedStateActionTable.loc[:, 'nextYVelSuccess'] = np.where(condensedStateActionTable['landingTypeSuccess'] == '#', 0, condensedStateActionTable['yVelAdj'])
        condensedStateActionTable.loc[:, 'nextXVelFail'] = np.where(condensedStateActionTable['landingTypeFail'] == '#', 0, condensedStateActionTable['xVel'])
        condensedStateActionTable.loc[:, 'nextYVelFail'] = np.where(condensedStateActionTable['landingTypeFail'] == '#', 0, condensedStateActionTable['yVel'])

        # merge back into the full state action table to calculate our outcomes
        fullStateActionTable = fullStateActionTable.merge(condensedStateActionTable, on=['xLoc', 'yLoc', 'xVel', 'yVel', 'xVelAdj', 'yVelAdj'])

        # subset by only the columns needed
        fullStateActionTable = fullStateActionTable[['xLoc', 'yLoc', 'xVel', 'yVel', 'xVelAdj', 'yVelAdj', 'xAccel', 'yAccel',
                                                     'nextXSuccess', 'nextYSuccess', 'landingTypeSuccess',
                                                     'nextXFail', 'nextYFail', 'landingTypeFail',
                                                     'nextXVelSuccess', 'nextYVelSuccess', 'nextXVelFail', 'nextYVelFail']]

        # init our Q value and times visited
        fullStateActionTable['QValue'] = 0

        print("**Merge in all of the relevant index maps " + self.trackType + " " + datetime.now().strftime("%d.%m.%Y_%I.%M.%S") + "**")

        # merge in our mapped values to the state in question for the current state
        fullStateActionTable = fullStateActionTable.merge(self.stateTable[['xLocState', 'yLocState', 'xVelState', 'yVelState', 'indexMap', 'locTypeState']],
                                                          left_on=['xLoc', 'yLoc', 'xVel', 'yVel'],
                                                          right_on=['xLocState', 'yLocState', 'xVelState', 'yVelState'],
                                                          how='left')
        fullStateActionTable = fullStateActionTable.drop(columns=['xLocState', 'yLocState', 'xVelState', 'yVelState'])
        fullStateActionTable = fullStateActionTable.rename(columns={'indexMap': 'currentStateValueMap', 'locTypeState': 'landingTypeCurrent'})

        # merge in our mapped values to the state in question for the next state if success
        fullStateActionTable = fullStateActionTable.merge(self.stateTable[['xLocState', 'yLocState', 'xVelState', 'yVelState', 'indexMap']],
                                                          left_on=['nextXSuccess', 'nextYSuccess', 'nextXVelSuccess', 'nextYVelSuccess'],
                                                          right_on=['xLocState', 'yLocState', 'xVelState', 'yVelState'],
                                                          how='left')
        fullStateActionTable = fullStateActionTable.drop(columns=['xLocState', 'yLocState', 'xVelState', 'yVelState'])
        fullStateActionTable = fullStateActionTable.rename(columns={'indexMap': 'successValueMap'})

        # merge in our mapped values to the state in question for the next state if fail
        fullStateActionTable = fullStateActionTable.merge(self.stateTable[['xLocState', 'yLocState', 'xVelState', 'yVelState', 'indexMap']],
                                                          left_on=['nextXFail', 'nextYFail', 'nextXVelFail', 'nextYVelFail'],
                                                          right_on=['xLocState', 'yLocState', 'xVelState', 'yVelState'],
                                                          how='left')
        fullStateActionTable = fullStateActionTable.drop(columns=['xLocState', 'yLocState', 'xVelState', 'yVelState'])
        fullStateActionTable = fullStateActionTable.rename(columns={'indexMap': 'failValueMap'})

        # write the table to memory for easy access next time
        fullStateActionTable.to_csv(stateActionTableFilePath, index=True)
        return fullStateActionTable

    def updateQValues(self):
        # merge in the success value from state table
        self.actionTable = self.actionTable.merge(self.stateTable[['indexMap', 'currentValue']], left_on=['successValueMap'], right_on=['indexMap'])
        self.actionTable = self.actionTable.drop(columns=['indexMap'])
        self.actionTable = self.actionTable.rename(columns={'currentValue': 'successValue'})
        self.actionTable['successValue'].fillna(0, inplace=True)

        # merge in the failure value from state table
        self.actionTable = self.actionTable.merge(self.stateTable[['indexMap', 'currentValue']], left_on=['failValueMap'], right_on=['indexMap'])
        self.actionTable = self.actionTable.drop(columns=['indexMap'])
        self.actionTable = self.actionTable.rename(columns={'currentValue': 'failValue'})
        self.actionTable['failValue'].fillna(0, inplace=True)

        # calculate the Q value
        self.actionTable.loc[:, 'QValue'] = - 1 + self.discountFactor * (.8 * self.actionTable['successValue'] + .2 * self.actionTable['failValue'])

        # drop the success/failure values to
        self.actionTable = self.actionTable.drop(columns=['successValue', 'failValue'])

    def updateValueTable(self):
        # save down the previous values for historical record
        nextEpochNumber = len(self.historicalValues)
        currentValues = self.stateTable['currentValue'].abs().sum()

        newRow = pd.DataFrame({'EpochNumber': [nextEpochNumber],
                               'SumOfValues': [currentValues]})

        self.historicalValues = pd.concat([self.historicalValues, newRow])

        # find the max Q value
        maxQAction = self.actionTable[['currentStateValueMap', 'QValue']].groupby(['currentStateValueMap'], as_index=False).max('QValue')

        # merge back into the state table
        self.stateTable = self.stateTable.merge(maxQAction,
                                                left_on=['indexMap'],
                                                right_on=['currentStateValueMap'],
                                                how='left')

        self.stateTable['QValue'].fillna(self.stateTable['currentValue'], inplace=True)
        self.stateTable.loc[:, 'currentValue'] = self.stateTable['QValue']
        self.stateTable = self.stateTable.drop(columns=['currentStateValueMap', 'QValue'])

    def checkConvergence(self):
        if len(self.historicalValues) == 1:
            return 1000

        # Get the last two values from column 'A'
        lastTwoValuesToCheck = self.historicalValues['SumOfValues'].iloc[-2:]

        # Calculate the difference
        convergenceCheck = lastTwoValuesToCheck.iloc[1] - lastTwoValuesToCheck.iloc[0]

        return convergenceCheck

    def printCurrentMap(self):
        # grab current raw track
        trackVisual = self.rawTrack

        # Determine the width of each cell for proper alignment
        max_val_len = len(str(max(max(row) for row in trackVisual)))
        index_width = max(len(str(len(trackVisual) - 1)), len(str(len(trackVisual[0]) - 1)))

        # Print column indices
        print(" " * (index_width + 4) + " ".join(f"{i:{max_val_len}}" for i in range(len(trackVisual[0]))))

        # Print the array rows with row indices
        for i, row in enumerate(trackVisual):
            print(f"{i:{index_width}} [" + " ".join(f"{x:{max_val_len}}" for x in row) + f"] {i:{index_width}}")

        # Print row indices at the bottom
        print(" " * (index_width + 2) + " ".join(f"{i:{max_val_len}}" for i in range(len(trackVisual[0]))))
