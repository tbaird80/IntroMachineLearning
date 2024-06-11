from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd


def dataSourcing(dataName):
    if dataName == 'BreastCancer':
        # fetch dataset
        breast_cancer_wisconsin_original = fetch_ucirepo(id=15)

        # data (as pandas dataframes)
        dataFeatures = pd.DataFrame(breast_cancer_wisconsin_original.data.features)
        dataTargets = pd.DataFrame(breast_cancer_wisconsin_original.data.targets)

        # remove NAs
        dataFeatures = dataFeatures.dropna()

        # assign types to all columns
        dataFeatures.loc[:, 'Clump_thickness'] = dataFeatures['Clump_thickness'].astype(float)
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
        # fetch dataset
        car_evaluation = fetch_ucirepo(id=19)

        # store relevant data (as pandas dataframes)
        dataFeatures = pd.DataFrame(car_evaluation.data.features)
        dataTargets = pd.DataFrame(car_evaluation.data.targets)

        # assign types to all columns
        dataFeatures = pd.get_dummies(dataFeatures).astype(bool)

        # set the name of the target column
        dataTargets.columns = ['Class']

    elif dataName == 'CongressVoting':
        # fetch dataset
        congressional_voting_records = fetch_ucirepo(id=105)

        # store relevant data (as pandas dataframes)
        dataFeatures = pd.DataFrame(congressional_voting_records.data.features)
        dataTargets = pd.DataFrame(congressional_voting_records.data.targets)

        # reset the data to be integers. helps to account for abstain votes which will be 0's
        dataFeatures = dataFeatures.replace({'y': 1, 'n': -1})
        dataFeatures = dataFeatures.fillna(0)

        # assign types to all columns
        dataFeatures.loc[:, 'handicapped-infants'] = dataFeatures['handicapped-infants'].astype(int)
        dataFeatures.loc[:, 'water-project-cost-sharing'] = dataFeatures['water-project-cost-sharing'].astype(int)
        dataFeatures.loc[:, 'adoption-of-the-budget-resolution'] = dataFeatures[
            'adoption-of-the-budget-resolution'].astype(int)
        dataFeatures.loc[:, 'physician-fee-freeze'] = dataFeatures['physician-fee-freeze'].astype(int)
        dataFeatures.loc[:, 'el-salvador-aid'] = dataFeatures['el-salvador-aid'].astype(int)
        dataFeatures.loc[:, 'religious-groups-in-schools'] = dataFeatures['religious-groups-in-schools'].astype(int)
        dataFeatures.loc[:, 'anti-satellite-test-ban'] = dataFeatures['anti-satellite-test-ban'].astype(int)
        dataFeatures.loc[:, 'aid-to-nicaraguan-contras'] = dataFeatures['aid-to-nicaraguan-contras'].astype(int)
        dataFeatures.loc[:, 'mx-missile'] = dataFeatures['mx-missile'].astype(int)
        dataFeatures.loc[:, 'immigration'] = dataFeatures['immigration'].astype(int)
        dataFeatures.loc[:, 'synfuels-corporation-cutback'] = dataFeatures['synfuels-corporation-cutback'].astype(int)
        dataFeatures.loc[:, 'education-spending'] = dataFeatures['education-spending'].astype(int)
        dataFeatures.loc[:, 'superfund-right-to-sue'] = dataFeatures['superfund-right-to-sue'].astype(int)
        dataFeatures.loc[:, 'crime'] = dataFeatures['crime'].astype(int)
        dataFeatures.loc[:, 'duty-free-exports'] = dataFeatures['duty-free-exports'].astype(int)
        dataFeatures.loc[:, 'export-administration-act-south-africa'] = dataFeatures[
            'export-administration-act-south-africa'].astype(int)

        # set the name of the target column
        dataTargets = dataTargets.loc[dataFeatures.index.tolist()]
        dataTargets.columns = ['Class']

    elif dataName == 'Abalone':
        # fetch dataset
        abalone = fetch_ucirepo(id=1)

        # store relevant data (as pandas dataframes)
        dataFeatures = pd.DataFrame(abalone.data.features)
        dataTargets = pd.DataFrame(abalone.data.targets)

        # assign types to all columns
        # performing one hot encoding for the sex column and then dropping the original column
        dataFeatures_Sex = dataFeatures['Sex']
        dataFeatures_Sex = pd.get_dummies(dataFeatures_Sex).astype(bool)
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
        # fetch dataset
        computer_hardware = fetch_ucirepo(id=29)

        # data (as pandas dataframes)
        dataFeatures = pd.DataFrame(computer_hardware.data.features)
        dataTargets = pd.DataFrame(dataFeatures['PRP'])

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
        # fetch dataset
        forest_fires = fetch_ucirepo(id=162)

        # data (as pandas dataframes)
        dataFeatures = pd.DataFrame(forest_fires.data.features)
        dataTargets = pd.DataFrame(forest_fires.data.targets)

        # assign types to all columns
        # setting these to integers as they are ordinal values
        dataFeatures.loc[:, 'X'] = dataFeatures['X'].astype(int)
        dataFeatures.loc[:, 'Y'] = dataFeatures['Y'].astype(int)

        # adjust these date values to relative integer values
        month_dict = {
            'jan': 1,
            'feb': 2,
            'mar': 3,
            'apr': 4,
            'may': 5,
            'jun': 6,
            'jul': 7,
            'aug': 8,
            'sep': 9,
            'oct': 10,
            'nov': 11,
            'dec': 12
        }
        # Replace month names with integer values
        dataFeatures.loc[:, 'month'] = dataFeatures['month'].replace(month_dict)
        dataFeatures.loc[:, 'month'] = dataFeatures['X'].astype(int)

        day_dict = {
            'mon': 1,
            'tue': 2,
            'wed': 3,
            'thu': 4,
            'fri': 5,
            'sat': 6,
            'sun': 7
        }
        # Replace month names with integer values
        dataFeatures.loc[:, 'day'] = dataFeatures['day'].replace(day_dict)
        dataFeatures.loc[:, 'day'] = dataFeatures['day'].astype(int)

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

    return dataFeatures, dataTargets

