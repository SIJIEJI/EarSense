from pathlib import Path
import pandas as pd

from Stress_study_ear_sum.dataInterface.featureOrganizer import FeatureOrganizer
from Stress_study_ear_sum.dataInterface.rawDataLoader import StressDataLoader


class FeatureSaver:
    def __init__(self, dataRoot, outputFolder='SavedFeatures'):
        self.dataRoot = Path(dataRoot)
        self.outputPath = self.dataRoot / outputFolder
        self.outputPath.mkdir(exist_ok=True)
        self.loader = StressDataLoader(dataRoot)
        self.organizer = FeatureOrganizer()

    def saveSubjectFeatures(self, subjectID, skip_existing=True):
        filepath = self.outputPath / f'subject{subjectID}_features.xlsx'
        # Skip if file already exists
        if skip_existing and filepath.exists():
            print(f'Already processed: {filepath}')
            return filepath

        subjectData = self.loader.loadSingleSubject(subjectID)
        featureDF = self.organizer.extractSubjectFeatures(subjectData, subjectID)
        if len(featureDF) == 0:
            print(f'No features to save for subject {subjectID}')
            return None
        featureDF.to_excel(filepath, index=False)
        print(f'Saved: {filepath}')
        return filepath

    def saveAllSubjects(self, subjects=None):
        if subjects is None:
            subjects = self.loader.subjects
        savedFiles = []
        for subjectID in subjects:
            print(f"Processing subject {subjectID}")
            filepath = self.saveSubjectFeatures(subjectID)
            if filepath:
                savedFiles.append(filepath)
        return savedFiles

    def saveCombinedFeatures(self, subjects=None, filename='all_subjects_features.xlsx'):
        if subjects is None:
            subjects = self.loader.subjects
        allDFs = []
        for subjectID in subjects:
            print(f"Processing subject {subjectID}")
            subjectdata = self.loader.loadSingleSubject(subjectID)
            featureDF = self.organizer.extractSubjectFeatures(subjectdata, subjectID)
            if len(featureDF) > 0:
                allDFs.append(featureDF)
        if not allDFs:
            print('No features extracted')
            return None
        combinedDF = pd.concat(allDFs, ignore_index=True)
        filepath = self.outputPath / filename
        combinedDF.to_excel(filepath, index=False)
        print(f' Saved: {filepath}')
        return filepath


if __name__ == "__main__":
      dataRoot = Path(__file__).parent.parent / 'Stress Data'
      saver = FeatureSaver(dataRoot)
      saver.saveAllSubjects()
