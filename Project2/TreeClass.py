import pandas as pd
import numpy as np
import AuxML2 as aux

class Tree:
    def __init__(self, dataName, isRegression, featuresMap, dataSet=pd.DataFrame(), existingTree=pd.DataFrame()):
        if len(existingTree) == 0:
            # initialize our data frame
            self.createTree(dataSet)
        else:
            self.treeTable = existingTree

        self.dataSetName = dataName
        self.featuresTypeMap = featuresMap
        self.isReg = isRegression

    def createTree(self, dataSet):
        treeOutput = pd.DataFrame()

        allFeatures = dataSet.columns

        nextFeature = aux.maxGainRatio(dataSet, allFeatures)

        treeOutput.addNode()


        self.treeTable = treeOutput

    def addNode(self, ):
        newRow = pd.DataFrame({
            'nodeFeature': [],
            'nodePrediction': [],
            'isLeaf': [],
            'childrenNodes': [{}],
            'decisionType': [],
            'dataSubset': []
        })

        treeOutput.concat(newRow)






