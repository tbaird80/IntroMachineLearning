import DataML2
from datetime import datetime
import os
import pandas as pd
from TreeClass import Tree
import AuxML2 as aux

if __name__ == '__main__':

    # function inputs
    # data set title
    dataTitle = 'LectureTest'

    # Define the table data
    data = {
        "Example": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        "Outlook": ["Sunny", "Sunny", "Overcast", "Rainy", "Rainy", "Rainy", "Overcast", "Sunny", "Sunny", "Rainy", "Sunny", "Overcast", "Overcast", "Rainy"],
        "Temperature": ["Hot", "Hot", "Hot", "Mild", "Cool", "Cool", "Cool", "Mild", "Cool", "Mild", "Mild", "Mild", "Hot", "Mild"],
        "Humidity": ["High", "High", "High", "High", "Normal", "Normal", "Normal", "High", "Normal", "Normal", "Normal", "High", "Normal", "High"],
        "Wind": [False, True, False, False, False, True, True, False, False, False, True, True, False, True],
        "Class": ["N", "N", "P", "P", "P", "N", "P", "N", "P", "P", "P", "P", "P", "N"]
    }

    # Create the DataFrame
    dataSet = pd.DataFrame(data)

    # define the features to be tuned
    featuresMap = {'Outlook': 'Cat',
                   'Temperature': 'Cat',
                   'Humidity': 'Cat',
                   'Wind': 'Cat'
                   }

    # define whether it is a regression
    regression = False

    # ----------------------Create Tree-----------------------------
    # Get the current timestamp and create our own unique new directory
    currentTimestamp = datetime.now()
    timestampStr = currentTimestamp.strftime("%d.%m.%Y_%I.%M.%S")
    uniqueTestID = dataTitle + "/" + timestampStr
    os.makedirs(uniqueTestID)

    # pruneSet, crossValidationSet = aux.splitDataFrame(dataSet=dataSet, splitPercentage=.2, isReg=regression)
    # trainSet, testSet = aux.splitDataFrame(dataSet=crossValidationSet, splitPercentage=.5, isReg=regression)

    prePruneTree = Tree(dataName=dataTitle, isRegression=regression, featuresMap=featuresMap, dataSet=dataSet)

    currentTreeID = 1
    currentTreeData = uniqueTestID + "/Tree" + str(currentTreeID)
    os.makedirs(currentTreeData)

    treeFileName = currentTreeData + "/prePruneTree.csv"
    trainDataFileName = currentTreeData + "/trainData.csv"
    # testDataFileName = currentTreeData + "/testData.csv"
    # pruneDataFileName = currentTreeData + "/pruneData.csv"

    prePruneTree.treeTable.to_csv(treeFileName, index=True)
    dataSet.to_csv(trainDataFileName, index=True)
    # testSet.to_csv(testDataFileName, index=True)
