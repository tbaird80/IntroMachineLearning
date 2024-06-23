import DataML2
from datetime import datetime
import os
import pandas as pd

if __name__ == '__main__':
    '''
    This is our main function for the Congress Voting test set. It will define, prune, and test the set to return an effectiveness
    value for the decision tree algorithm. If you would like to run yourself, I would recommend doing so in chunks. Create full tree first,
    prune tree, test full tree, test prune tree, and then compare results.

    '''
    # function inputs
    # data set title
    dataTitle = 'CongressVoting'

    # grab data
    features, targets = DataML2.dataSourcing(dataTitle)
    dataSet = features.join(targets)

    # define the features to be tuned
    featuresMap = {
        'handicapped-infants': 'Num',
        'water-project-cost-sharing': 'Num',
        'adoption-of-the-budget-resolution': 'Num',
        'physician-fee-freeze': 'Num',
        'el-salvador-aid': 'Num',
        'religious-groups-in-schools': 'Num',
        'anti-satellite-test-ban': 'Num',
        'aid-to-nicaraguan-contras': 'Num',
        'mx-missile': 'Num',
        'immigration': 'Num',
        'synfuels-corporation-cutback': 'Num',
        'education-spending': 'Num',
        'superfund-right-to-sue': 'Num',
        'crime': 'Num',
        'duty-free-exports': 'Num',
        'export-administration-act-south-africa': 'Num'
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