import numpy as np
from ucimlrepo import fetch_ucirepo
import pandas as pd
import os

def dataSourcing(dataName):
    """
    Source the relevant dataset either locally or remotely depending on it has been sourced previously.

    @param dataName: the data set that we would like to source
    @return: the data set features and the data set targets
    """

    featurePath = dataName + "/featureData.csv"
    targetPath = dataName + "/targetData.csv"

    if dataName == 'BreastCancer':
        # check if file path exists, read from file if so, otherwise grab from online repo
        if os.path.exists(featurePath) & os.path.exists(targetPath):
            print(dataName + " data exists, reading from csv")

            # read from local directory
            dataFeatures = pd.read_csv(featurePath, index_col=0)
            dataTargets = pd.read_csv(targetPath, index_col=0)

        else:
            print(dataName + "data does not exist, reading from source")

            # fetch dataset
            breast_cancer_wisconsin_original = fetch_ucirepo(id=15)

            # data (as pandas dataframes)
            dataFeatures = pd.DataFrame(breast_cancer_wisconsin_original.data.features)
            dataTargets = pd.DataFrame(breast_cancer_wisconsin_original.data.targets)

            # write to local directory
            dataFeatures.to_csv(featurePath, index=True)
            dataTargets.to_csv(targetPath, index=True)

        # remove NAs
        dataFeatures = dataFeatures.dropna()

        # assign types to all columns
        dataFeatures.loc[:, 'Clump_thickness'] = dataFeatures['Clump_thickness'].astype(float)
        dataFeatures.loc[:, 'Uniformity_of_cell_size'] = dataFeatures['Uniformity_of_cell_size'].astype(float)
        dataFeatures.loc[:, 'Uniformity_of_cell_shape'] = dataFeatures['Uniformity_of_cell_shape'].astype(float)
        dataFeatures.loc[:, 'Marginal_adhesion'] = dataFeatures['Marginal_adhesion'].astype(float)
        dataFeatures.loc[:, 'Single_epithelial_cell_size'] = dataFeatures['Single_epithelial_cell_size'].astype(float)
        dataFeatures.loc[:, 'Bare_nuclei'] = dataFeatures['Bare_nuclei'].astype(float)
        dataFeatures.loc[:, 'Bland_chromatin'] = dataFeatures['Bland_chromatin'].astype(float)
        dataFeatures.loc[:, 'Normal_nucleoli'] = dataFeatures['Normal_nucleoli'].astype(float)
        dataFeatures.loc[:, 'Mitoses'] = dataFeatures['Mitoses'].astype(float)

        # set the name of the target column
        dataTargets = dataTargets.loc[dataFeatures.index.tolist()]
        dataTargets.columns = ['Class']

    elif dataName == 'CarEval':
        # check if file path exists, read from file if so, otherwise grab from online repo
        if os.path.exists(featurePath) & os.path.exists(targetPath):
            print(dataName + " data exists, reading from csv")

            # read from local directory
            dataFeatures = pd.read_csv(featurePath, index_col=0)
            dataTargets = pd.read_csv(targetPath, index_col=0)

        else:
            print(dataName + " data does not exist, reading from source")

            # fetch dataset
            car_evaluation = fetch_ucirepo(id=19)

            # store relevant data (as pandas dataframes)
            dataFeatures = pd.DataFrame(car_evaluation.data.features)
            dataTargets = pd.DataFrame(car_evaluation.data.targets)

            # write to the local directory
            dataFeatures.to_csv(featurePath, index=True)
            dataTargets.to_csv(targetPath, index=True)

        # adjust these categorical values to relative integer values
        buying_dict = {
            'low': 0,
            'med': 1,
            'high': 2,
            'vhigh': 3
        }
        # replace names with integer values
        dataFeatures.loc[:, 'buying'] = dataFeatures['buying'].replace(buying_dict)
        dataFeatures.loc[:, 'buying'] = dataFeatures['buying'].astype(int)

        # adjust these categorical values to relative integer values
        maint_dict = {
            'low': 0,
            'med': 1,
            'high': 2,
            'vhigh': 3
        }
        # replace names with integer values
        dataFeatures.loc[:, 'maint'] = dataFeatures['maint'].replace(maint_dict)
        dataFeatures.loc[:, 'maint'] = dataFeatures['maint'].astype(int)

        # adjust these categorical values to relative integer values
        doors_dict = {
            '2': 0,
            '3': 1,
            '4': 2,
            '5more': 3
        }
        # replace names with integer values
        dataFeatures.loc[:, 'doors'] = dataFeatures['doors'].replace(doors_dict)
        dataFeatures.loc[:, 'doors'] = dataFeatures['doors'].astype(int)

        # adjust these categorical values to relative integer values
        persons_dict = {
            '2': 0,
            '4': 1,
            'more': 2
        }
        # replace names with integer values
        dataFeatures.loc[:, 'persons'] = dataFeatures['persons'].replace(persons_dict)
        dataFeatures.loc[:, 'persons'] = dataFeatures['persons'].astype(int)

        # adjust these categorical values to relative integer values
        lugBoot_dict = {
            'small': 0,
            'med': 1,
            'big': 2
        }
        # replace names with integer values
        dataFeatures.loc[:, 'lug_boot'] = dataFeatures['lug_boot'].replace(lugBoot_dict)
        dataFeatures.loc[:, 'lug_boot'] = dataFeatures['lug_boot'].astype(int)

        # adjust these categorical values to relative integer values
        safety_dict = {
            'low': 0,
            'med': 1,
            'high': 2
        }
        # replace names with integer values
        dataFeatures.loc[:, 'safety'] = dataFeatures['safety'].replace(safety_dict)
        dataFeatures.loc[:, 'safety'] = dataFeatures['safety'].astype(int)

        # set the name of the target column
        dataTargets.columns = ['Class']

    elif dataName == 'CongressVoting':
        # check if file path exists, read from file if so, otherwise grab from online repo
        if os.path.exists(featurePath) & os.path.exists(targetPath):
            print(dataName + " data exists, reading from csv")

            # read from local directory
            dataFeatures = pd.read_csv(featurePath, index_col=0)
            dataTargets = pd.read_csv(targetPath, index_col=0)

        else:
            print(dataName + " data does not exist, reading from source")

            # fetch dataset
            congressional_voting_records = fetch_ucirepo(id=105)

            # store relevant data (as pandas dataframes)
            dataFeatures = pd.DataFrame(congressional_voting_records.data.features)
            dataTargets = pd.DataFrame(congressional_voting_records.data.targets)

            # write to the local directory
            dataFeatures.to_csv(featurePath, index=True)
            dataTargets.to_csv(targetPath, index=True)

        # reset the data to be integers. helps to account for abstain votes which will be 0's
        dataFeatures = dataFeatures.replace({'y': 2, 'n': 0})
        dataFeatures = dataFeatures.fillna(1)

        dataFeatures = dataFeatures.replace({2: 'yes', 1: 'abstain', 0: 'no'})

        # assign types to all columns
        dataFeatures = pd.get_dummies(dataFeatures).astype(int)

        # set the name of the target column
        dataTargets = dataTargets.loc[dataFeatures.index.tolist()]
        dataTargets.columns = ['Class']

    elif dataName == 'Abalone':
        # check if file path exists, read from file if so, otherwise grab from online repo
        if os.path.exists(featurePath) & os.path.exists(targetPath):
            print(dataName + " data exists, reading from csv")

            # read from local directory
            dataFeatures = pd.read_csv(featurePath, index_col=0)
            dataTargets = pd.read_csv(targetPath, index_col=0)

        else:
            print(dataName + " data does not exist, reading from source")

            # fetch dataset
            abalone = fetch_ucirepo(id=1)

            # store relevant data (as pandas dataframes)
            dataFeatures = pd.DataFrame(abalone.data.features)
            dataTargets = pd.DataFrame(abalone.data.targets)

            # write to the local directory
            dataFeatures.to_csv(featurePath, index=True)
            dataTargets.to_csv(targetPath, index=True)

        # assign types to all columns
        # performing one hot encoding for the sex column and then dropping the original column
        dataFeatures_Sex = dataFeatures['Sex']
        dataFeatures_Sex = pd.get_dummies(dataFeatures_Sex).astype(int)
        dataFeatures = dataFeatures.join(dataFeatures_Sex)
        dataFeatures = dataFeatures.drop(columns='Sex')

        # normalize the rest
        dataFeatures.loc[:, 'Length'] = dataFeatures['Length'].astype(float)
        dataFeatures.loc[:, 'Diameter'] = dataFeatures['Diameter'].astype(float)
        dataFeatures.loc[:, 'Height'] = dataFeatures['Height'].astype(float)
        dataFeatures.loc[:, 'Whole_weight'] = dataFeatures['Whole_weight'].astype(float)
        dataFeatures.loc[:, 'Shucked_weight'] = dataFeatures['Shucked_weight'].astype(float)
        dataFeatures.loc[:, 'Viscera_weight'] = dataFeatures['Viscera_weight'].astype(float)
        dataFeatures.loc[:, 'Shell_weight'] = dataFeatures['Shell_weight'].astype(float)

        # set the name of the target column
        dataTargets = dataTargets.loc[dataFeatures.index.tolist()]
        dataTargets.columns = ['Class']

    elif dataName == 'ComputerHardware':
        # check if file path exists, read from file if so, otherwise grab from online repo
        if os.path.exists(featurePath) & os.path.exists(targetPath):
            print(dataName + "data exists, reading from csv")

            # read from local directory
            dataFeatures = pd.read_csv(featurePath, index_col=0)
            dataTargets = pd.read_csv(targetPath, index_col=0)

        else:
            print(dataName + "data does not exist, reading from source")

            # fetch dataset
            computer_hardware = fetch_ucirepo(id=29)

            # data (as pandas dataframes)
            dataFeatures = pd.DataFrame(computer_hardware.data.features)
            dataTargets = pd.DataFrame(dataFeatures['PRP'])

            # write to the local directory
            dataFeatures.to_csv(featurePath, index=True)
            dataTargets.to_csv(targetPath, index=True)

        # drop the columns that we do not need
        dataFeatures = dataFeatures.drop(columns=['VendorName', 'ModelName', 'PRP', 'ERP'])

        # assign types to all columns, normalizing these
        dataFeatures.loc[:, 'MYCT'] = dataFeatures['MYCT'].astype(float)
        dataFeatures.loc[:, 'MMIN'] = dataFeatures['MMIN'].astype(float)
        dataFeatures.loc[:, 'MMAX'] = dataFeatures['MMAX'].astype(float)
        dataFeatures.loc[:, 'CACH'] = dataFeatures['CACH'].astype(float)
        dataFeatures.loc[:, 'CHMIN'] = dataFeatures['CHMIN'].astype(float)
        dataFeatures.loc[:, 'CHMAX'] = dataFeatures['CHMAX'].astype(float)

        # set the name of the target column
        dataTargets = dataTargets.loc[dataFeatures.index.tolist()]
        dataTargets.columns = ['Class']

    elif dataName == 'ForestFires':
        # check if file path exists, read from file if so, otherwise grab from online repo
        if os.path.exists(featurePath) & os.path.exists(targetPath):
            print(dataName + "data exists, reading from csv")

            # read from local directory
            dataFeatures = pd.read_csv(featurePath, index_col=0)
            dataTargets = pd.read_csv(targetPath, index_col=0)

        else:
            print(dataName + "data does not exist, reading from source")

            # fetch dataset
            forest_fires = fetch_ucirepo(id=162)

            # data (as pandas dataframes)
            dataFeatures = pd.DataFrame(forest_fires.data.features)
            dataTargets = pd.DataFrame(forest_fires.data.targets)

            # write to the local directory
            dataFeatures.to_csv(featurePath, index=True)
            dataTargets.to_csv(targetPath, index=True)

        # assign types to all columns
        # setting these to integers as they are ordinal values
        dataFeatures.loc[:, 'X'] = dataFeatures['X'].astype(int)
        dataFeatures.loc[:, 'Y'] = dataFeatures['Y'].astype(int)

        # set month as one hot encoding
        dataFeatures_Month = dataFeatures['month']
        dataFeatures_Month = pd.get_dummies(dataFeatures_Month).astype(int)
        dataFeatures = dataFeatures.join(dataFeatures_Month)
        dataFeatures = dataFeatures.drop(columns='month')

        # set day as one hot encoding
        dataFeatures_Day = dataFeatures['day']
        dataFeatures_Day = pd.get_dummies(dataFeatures_Day).astype(int)
        dataFeatures = dataFeatures.join(dataFeatures_Day)
        dataFeatures = dataFeatures.drop(columns='day')

        # normalize these
        dataFeatures.loc[:, 'FFMC'] = dataFeatures['FFMC'].astype(float)
        dataFeatures.loc[:, 'DMC'] = dataFeatures['DMC'].astype(float)
        dataFeatures.loc[:, 'DC'] = dataFeatures['DC'].astype(float)
        dataFeatures.loc[:, 'ISI'] = dataFeatures['ISI'].astype(float)
        dataFeatures.loc[:, 'temp'] = dataFeatures['temp'].astype(float)
        dataFeatures.loc[:, 'RH'] = dataFeatures['RH'].astype(float)
        dataFeatures.loc[:, 'wind'] = dataFeatures['wind'].astype(float)
        dataFeatures.loc[:, 'rain'] = dataFeatures['rain'].astype(float)

        # set the name of the target column
        dataTargets = dataTargets.loc[dataFeatures.index.tolist()]
        dataTargets.columns = ['Class']
        dataTargets.loc[:, 'Class'] = np.log1p(dataTargets['Class'])

    return dataFeatures, dataTargets

