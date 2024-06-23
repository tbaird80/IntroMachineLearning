# Define the table data

import DataML2
from datetime import datetime
import os
import pandas as pd

if __name__ == '__main__':
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
