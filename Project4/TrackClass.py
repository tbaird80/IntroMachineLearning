import AuxML4 as Aux
import pandas as pd
import numpy as np
import datetime

class Track:

    def __init__(self, trackType, learnType, learningRate, discountFactor=0, epsilon=.1):
        self.currentTrack = Aux.readTrackFile(trackType)
        self.learnType = learnType
        self.learningRate = learningRate
        self.discountFactor = discountFactor
        self.epsilon = epsilon

        print("***********************Starting to build State Table for " + trackType + " " + datetime.now().strftime("%d.%m.%Y_%I.%M.%S") + "*****************************")
        self.stateTable = self.createStateTable()

        print("***********************Starting to build Action Table for " + trackType + " " + datetime.now().strftime("%d.%m.%Y_%I.%M.%S") + "*****************************")
        self.actionTable = self.createActionTable()

    def createStateTable(self):
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
        fullStateTable.loc[:, 'nextValue'] = 0
        fullStateTable.loc[:, 'timesVisited'] = 0

        return fullStateTable

    def createActionTable(self):
        # find the coordinate points that are valid action spaces
        nonFinishCoordinates = self.currentTrack[self.currentTrack.locType.isin(['S', '.'])]

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
        fullStateActionTable = nonFinishCoordinates.merge(stateActionOptions, how='cross')

        # subset table as there are overlaps in state/action/result
        condensedStateActionTable = fullStateActionTable[['xLoc', 'yLoc', 'xVel', 'yVel', 'xVelAdj', 'yVelAdj']].drop_duplicates()

        # find the next possible location based on both a successful acceleration and not successful acceleration
        condensedStateActionTable.loc[:, 'xLocNextSuccess'] = condensedStateActionTable['xLoc'] + condensedStateActionTable['xVelAdj']
        condensedStateActionTable.loc[:, 'yLocNextSuccess'] = condensedStateActionTable['yLoc'] + condensedStateActionTable['yVelAdj']
        condensedStateActionTable.loc[:, 'xLocNextFail'] = condensedStateActionTable['xLoc'] + condensedStateActionTable['xVel']
        condensedStateActionTable.loc[:, 'yLocNextFail'] = condensedStateActionTable['yLoc'] + condensedStateActionTable['yVel']

        # find all the intermediate locations between proposed start and end locations on map
        condensedStateActionTable.loc[:, 'nextLocationsSuccess'] = condensedStateActionTable.apply(lambda row: Aux.bresenhamsAlgorithm(row['xLoc'],
                                                                                                                                       row['yLoc'],
                                                                                                                                       row['xLocNextSuccess'],
                                                                                                                                       row['yLocNextSuccess']), axis=1)
        condensedStateActionTable.loc[:, 'nextLocationsFail'] = condensedStateActionTable.apply(lambda row: Aux.bresenhamsAlgorithm(row['xLoc'],
                                                                                                                                    row['yLoc'],
                                                                                                                                    row['xLocNextFail'],
                                                                                                                                    row['yLocNextFail']), axis=1)

        # given the possible locations, provide the actual landing spot accounting for the shape of the track
        condensedStateActionTable.loc[:, 'landingSpotSuccess'] = condensedStateActionTable.apply(lambda row: Aux.nextTrackLoc(self.currentTrack, row['nextLocationsSuccess']), axis=1)
        condensedStateActionTable.loc[:, 'landingSpotFail'] = condensedStateActionTable.apply(lambda row: Aux.nextTrackLoc(self.currentTrack, row['nextLocationsFail']), axis=1)

        # condense the state action table further given overlaps to speed up calculations
        furtherCondensedStateActionTable = condensedStateActionTable[['landingSpotSuccess', 'landingSpotFail']].drop_duplicates()

        # pull out the relevant X/Y/type of the actual landing spot calculated above
        furtherCondensedStateActionTable.loc[:, 'nextXSuccess'] = furtherCondensedStateActionTable.apply(lambda row: row['landingSpotSuccess'][0], axis=1)
        furtherCondensedStateActionTable.loc[:, 'nextYSuccess'] = furtherCondensedStateActionTable.apply(lambda row: row['landingSpotSuccess'][1], axis=1)
        furtherCondensedStateActionTable.loc[:, 'landingTypeSuccess'] = furtherCondensedStateActionTable.apply(lambda row: row['landingSpotSuccess'][2], axis=1)
        furtherCondensedStateActionTable.loc[:, 'nextXFail'] = furtherCondensedStateActionTable.apply(lambda row: row['landingSpotFail'][0], axis=1)
        furtherCondensedStateActionTable.loc[:, 'nextYFail'] = furtherCondensedStateActionTable.apply(lambda row: row['landingSpotFail'][1], axis=1)
        furtherCondensedStateActionTable.loc[:, 'landingTypeFail'] = furtherCondensedStateActionTable.apply(lambda row: row['landingSpotFail'][2], axis=1)

        # merge back into condensed state once states pulled out
        condensedStateActionTable = condensedStateActionTable.merge(furtherCondensedStateActionTable, on=['landingSpotSuccess', 'landingSpotFail'])

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

        # init our Q value
        fullStateActionTable['QValue'] = 0

        # merge in our mapped values to the state in question for the current state
        fullStateActionTable = fullStateActionTable.merge(self.stateTable[['xLocState', 'yLocState', 'xVelState', 'yVelState', 'indexMap']],
                                                          left_on=['xLoc', 'yLoc', 'xVel', 'yVel'],
                                                          right_on=['xLocState', 'yLocState', 'xVelState', 'yVelState'],
                                                          how='left')
        fullStateActionTable = fullStateActionTable.drop(columns=['xLocState', 'yLocState', 'xVelState', 'yVelState'])
        fullStateActionTable = fullStateActionTable.rename(columns={'indexMap': 'currentStateValueMap'})

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

        return fullStateActionTable

    def updateQValues(self):
        if self.learnType == 'valueIteration':
            # merge in the success value from state table
            self.actionTable = self.actionTable.merge(self.stateTable[['indexMap', 'currentValue']], left_on=['successValueMap'], right_on=['indexMap'])
            self.actionTable = self.actionTable.drop(columns=['indexMap'])
            self.actionTable = self.actionTable.rename(columns={'currentValue': 'successValue'})

            # merge in the failure value from state table
            self.actionTable = self.actionTable.merge(self.stateTable[['indexMap', 'currentValue']], left_on=['failValueMap'], right_on=['indexMap'])
            self.actionTable = self.actionTable.drop(columns=['indexMap'])
            self.actionTable = self.actionTable.rename(columns={'currentValue': 'failValue'})

            # calculate the Q value
            self.actionTable.loc[:, 'QValue'] = .9 * (.8 * self.actionTable['successValue'] + .2 * self.actionTable['failValue'])

            # drop the success/failure values to
            self.actionTable = self.actionTable.drop(columns=['successValue', 'failValue'])

    def updateValueTable(self):
        # find the max Q value
        maxQAction = self.actionTable[['currentStateValueMap', 'QValue']].groupby(['currentStateValueMap'], as_index=False).max('QValue')

        self.stateTable = self.stateTable.merge(maxQAction,
                                                left_on=['indexMap'],
                                                right_on=['currentStateValueMap'],
                                                how='left')

        self.stateTable['QValue'].fillna(self.stateTable['nextValue'], inplace=True)
        self.stateTable.loc[:, 'nextValue'] = self.stateTable['QValue']
        self.stateTable = self.stateTable.drop(columns=['currentStateValueMap', 'QValue'])

