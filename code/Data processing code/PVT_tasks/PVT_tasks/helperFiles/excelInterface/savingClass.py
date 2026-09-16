import os
import numpy as np
import pandas as pd
from openpyxl import Workbook


class savingProtocol:
    def __init__(self, save_folder):
        self.saveFolder = save_folder
        if not os.path.exists(self.saveFolder):
            os.makedirs(self.saveFolder)

    def saveExperimentResults(self, filename, dataframes: dict):
        full_path = os.path.join(self.saveFolder, filename)
        with pd.ExcelWriter(full_path) as writer:
            for sheet_name, df in dataframes.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"Experiment data saved to {full_path}")


