import AuxML4 as Aux
import AuxTestML4 as AuxTest
import pandas as pd
import numpy as np
import TrackClass

if __name__ == '__main__':
    """
    Main function to help run the L track

    """

    trackType = 'L'  # replace with your file name

    learnTypeList = ['SARSA', 'QLearning']
    tauList = [100, 200]
    DFList = [.9, .99]

    for currentLearn in learnTypeList:
        for currentTau in tauList:
            for currentDF in DFList:
                for currentTrackID in range(2, 9):
                    currentTrackName = "Smaller" + str(currentTrackID) + trackType
                    nextTrack = TrackClass.Track(trackName=currentTrackName, trackFamily=trackType, learnType=currentLearn,
                                                 discountFactor=currentDF, tau=currentTau, smallerTrackID=currentTrackID)
                    Aux.trainQLearningSARSASubTrack(nextTrack, trainType=currentLearn)
                    currentTrackID += 1

                nextTrack = TrackClass.Track(trackName=trackType, trackFamily=trackType, learnType=currentLearn,
                                             discountFactor=currentDF, tau=currentTau, smallerTrackID=0)
                testRuns = AuxTest.runTests(nextTrack, trainType=currentLearn)

    nextTrack = TrackClass.Track(trackName=trackType, trackFamily=trackType, learnType='SARSA',
                                 discountFactor=.9, tau=200, smallerTrackID=0)
    testRuns = AuxTest.runTests(nextTrack, trainType='SARSA')

    nextTrack = TrackClass.Track(trackName=trackType, trackFamily=trackType, learnType='QLearning',
                                 discountFactor=.9, tau=100, smallerTrackID=0)
    testRuns = AuxTest.runTests(nextTrack, trainType='QLearning')

    DFList = [.9, .99]

    for currentDF in DFList:
        nextTrack = TrackClass.Track(trackName=trackType, trackFamily=trackType, learnType='ValueIteration',
                                     discountFactor=currentDF)
        AuxTest.runTestsVI(nextTrack)
