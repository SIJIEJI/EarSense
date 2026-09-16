
class Config:
    fileModalities = ['questionnaire', 'eeg_ecg', 'peripheral']
    signalsType = ['GSR', 'PPG', 'TEMP', 'ACC', 'ECG', 'EEG']
    # classLabels = {'CPT_Baseline': 0, 'CPT_Stressor': 1, 'CPT_Recovery': 2, 'VR_Baseline': 3, 'VR_Stressor': 4, 'VR_Recovery': 5}
    classLabels = {'CPT_Baseline': 0, 'CPT_Stressor': 1, 'VR_Stressor': 2, 'VR_Recovery': 3} # 4 classes
    condition_order = {'Baseline': 0, 'Stressor': 1, 'Recovery': 2}

    # Windowing parameters
    # global window size
    windowSize = 30
    windowStep = 30

    # look back configurations
    AccLookBack = 5
    EEGLookBack = 30
    ECGLookBack = 30
    PPGLookBack = 30
    GSRLookBack = 30
    TempLookBack = 30

    # filtering parameters
    notchFreq = 60
    ECGBandPass = (1, 50) # ECG
    EEGBandPass = (1, 50) # EEG
    PPGGreenPulseBandPass = (0.5, 5) # PPG green for pulse
    PPGGreenRespBandPass = (0.1, 0.8) # ppg green for respiration
    PPGIRBandPass = (0.5, 5) # ppg ir
    PPGRedBandPass = (0.5, 5) # ppg red

    # feature modality ablation
    # keys match featureOrganizer.featureNames; values are the feature-name prefixes
    modalityPrefixes = {'ACC': 'acc_', 'ECG': 'ecg_', 'EEG': 'eeg_',
                        'PPG': 'ppg_', 'GSR': 'gsr_', 'Temp': 'temp_'}
    excludeModalities = []  # e.g. ['ACC'] drops the IMU features

    baselineClasses = [0]
    centerRegressionTarget = True

    # target:
    regTargetNames = ['PA', 'NA', 'SA']
    regTargets = [2]  # indices into regTargetNames; e.g. [0, 1, 2] runs PA, NA and SA

    # split
    splitMethod = ['rolling_origin']  # ['rolling_origin', 'loso', 'sgkf', 'skf']
    rollingFolds = 4

    embargo = 1  # prevent overlap
    smoothWindow = 8
    runShap = True

    # output
    outputDir = './results'
    shapDir = './shapResults'
