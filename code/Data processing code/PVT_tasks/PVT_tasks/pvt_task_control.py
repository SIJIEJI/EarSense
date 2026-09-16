import os
import sys
from helperFiles.excelInterface.savingClass import savingProtocol
from experimentProtocols.sustainedVigilance.pureSustainedVigilanceTask import PVTTask



def getPureExperimentsToRun(experimentName, experimentModality):
    experiment_map = {
        "SustainedVigilance": {
            "PVT": PVTTask(total_duration=600, isi_range=(0.75, 1.5), response_window=2, num_blocks=200),
        }
    }
    return experiment_map[experimentName][experimentModality]


if __name__ == "__main__":
    settings = 'Lab' # Real
    date = "20251028"
    subjectName = "Christopher-patient"
    experimentToRun = 'SustainedVigilance'
    experimentModality = 'PVT' # PVT
    if settings == 'Lab':
        excelFolder = os.path.join(os.getcwd(), 'RealTimeExperimentFiles', f'{subjectName}_{date}')
        saveFileName = f'{subjectName}_{date}_{experimentToRun}_{experimentModality}_{settings}_results.xlsx'
        savingClass = savingProtocol(excelFolder)

        # running experiment
        experimentController = getPureExperimentsToRun(experimentToRun, experimentModality)
        experimentResults = experimentController.run()

        user_input = input("Data acquisition complete. Do you want to save the acquired data? (y/n): ")
        if user_input.strip().lower() == 'y':
            savingClass.saveExperimentResults(saveFileName, experimentResults)