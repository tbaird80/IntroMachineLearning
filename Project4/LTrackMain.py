import AuxML4 as Aux
import pandas as pd
import numpy as np
import TrackClass

if __name__ == '__main__':
    trackType = 'L'  # replace with your file name

    # newTrack = TrackClass.Track(trackType, learnType='valueIteration', learningRate=0, discountFactor=.9, epsilon=.1)

    # Aux.trainValueIteration(newTrack)

    # numSmallerTracks = 7
    # currentTrackID = 1
    #
    # while currentTrackID <= numSmallerTracks:
    #     currentTrackName = "Smaller" + str(currentTrackID) + trackType
    #     nextTrack = TrackClass.Track(currentTrackName, learnType='valueIteration', learningRate=0, discountFactor=.9, epsilon=.1)
    #     currentTrackID += 1

    currentTrackID = 1
    currentTrackName = "Smaller" + str(currentTrackID) + trackType
    nextTrack = TrackClass.Track(currentTrackName, learnType='QLearning', learningRate=0, discountFactor=.99, epsilon=.1)

    Aux.trainQLearningSARSASubTrack(nextTrack, trainType='Q')


