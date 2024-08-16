import AuxML4 as Aux
import AuxTestML4 as AuxTest
import pandas as pd
import numpy as np
import TrackClass

if __name__ == '__main__':
    trackType = 'W'  # replace with your file name

    learnTypeList = ['SARSA', 'QLearning']
    tauList = [100, 200]
    DFList = [.9, .99]

    for currentLearn in learnTypeList:
        for currentTau in tauList:
            for currentDF in DFList:
                for currentTrackID in range(1, 11):
                    currentTrackName = "Smaller" + str(currentTrackID) + trackType
                    nextTrack = TrackClass.Track(trackName=currentTrackName, trackFamily=trackType, learnType=currentLearn,
                                                 discountFactor=currentDF, tau=currentTau, smallerTrackID=currentTrackID)
                    Aux.trainQLearningSARSASubTrack(nextTrack, trainType=currentLearn)
                    currentTrackID += 1

                nextTrack = TrackClass.Track(trackName=trackType, trackFamily=trackType, learnType=currentLearn,
                                             discountFactor=currentDF, tau=currentTau, smallerTrackID=0)
                testRuns = AuxTest.runTests(nextTrack, trainType=currentLearn)

    # for currentTrackID in range(1, 6):
    #     currentTrackName = "Smaller" + str(currentTrackID) + trackType
    #     nextTrack = TrackClass.Track(trackName=currentTrackName, trackFamily=trackType, learnType='QLearning',
    #                                  discountFactor=.9, smallerTrackID=currentTrackID)
    #     Aux.trainQLearningSARSASubTrack(nextTrack, trainType='QLearning')
    #     currentTrackID += 1
    #
    # for currentTrackID in range(1, 11):
    #     currentTrackName = "Smaller" + str(currentTrackID) + trackType
    #     nextTrack = TrackClass.Track(trackName=currentTrackName, trackFamily=trackType, learnType='QLearning',
    #                                  discountFactor=.99, smallerTrackID=currentTrackID)
    #     Aux.trainQLearningSARSASubTrack(nextTrack, trainType='QLearning')
    #     currentTrackID += 1
    #
    # for currentTrackID in range(1, 6):
    #     currentTrackName = "Smaller" + str(currentTrackID) + trackType
    #     nextTrack = TrackClass.Track(trackName=currentTrackName, trackFamily=trackType, learnType='SARSA',
    #                                  discountFactor=.9, smallerTrackID=currentTrackID)
    #     Aux.trainQLearningSARSASubTrack(nextTrack, trainType='SARSA')
    #     currentTrackID += 1
    #
    # for currentTrackID in range(1, 11):
    #     currentTrackName = "Smaller" + str(currentTrackID) + trackType
    #     nextTrack = TrackClass.Track(trackName=currentTrackName, trackFamily=trackType, learnType='SARSA',
    #                                  discountFactor=.99, smallerTrackID=currentTrackID)
    #     Aux.trainQLearningSARSASubTrack(nextTrack, trainType='SARSA')
    #     currentTrackID += 1
    #
    # currentTrackName = trackType
    # nextTrack = TrackClass.Track(trackName=currentTrackName, trackFamily=trackType, learnType='QLearning',
    #                              discountFactor=.99, smallerTrackID=0)
    # testRuns = AuxTest.runTests(nextTrack, trainType='QLearning')
    #
    # currentTrackName = trackType
    # nextTrack = TrackClass.Track(trackName=currentTrackName, trackFamily=trackType, learnType='SARSA',
    #                              discountFactor=.99, smallerTrackID=0)
    # testRuns = AuxTest.runTests(nextTrack, trainType='SARSA')
    #
