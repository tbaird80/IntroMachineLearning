import DataML2
from datetime import datetime
import os
import pandas as pd

if __name__ == '__main__':
    '''
    This is our main function for the Breast Cancer test set. It will define, prune, and test the set to return an effectiveness
    value for the decision tree algorithm. If you would like to run yourself, I would recommend doing so in chunks. Create full tree first,
    prune tree, test full tree, test prune tree, and then compare results.

    '''
    # function inputs
    # data set title
    dataTitle = 'BreastCancer'

    # grab data
    features, targets = DataML2.dataSourcing(dataTitle)
    dataSet = features.join(targets)

    # define the features to be tuned
    featuresMap = {
        'Clump_thickness': 'Num',
        'Uniformity_of_cell_size': 'Num',
        'Uniformity_of_cell_shape': 'Num',
        'Marginal_adhesion': 'Num',
        'Single_epithelial_cell_size': 'Num',
        'Bare_nuclei': 'Num',
        'Bland_chromatin': 'Num',
        'Normal_nucleoli': 'Num',
        'Mitoses': 'Num'
    }

    # define whether it is a regression
    regression = False

    # ----------------------Create Tree-----------------------------
    # Get the current timestamp and create our own unique new directory
    currentTimestamp = datetime.now()
    timestampStr = currentTimestamp.strftime("%d.%m.%Y_%I.%M.%S")
    uniqueTestID = dataTitle + "/" + timestampStr
    os.makedirs(uniqueTestID)

    # -----------------------Prune Tree------------------------------

    # -----------------------Test Full Tree--------------------------

    # -----------------------Test Pruned Tree------------------------
