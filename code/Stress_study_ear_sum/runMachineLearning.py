import numpy as np
import os
from Stress_study_ear_sum.machineLearning.pipelines import MLPipeline
from Stress_study_ear_sum.plottingHelpers.plot_results import create_combined_report
from Stress_study_ear_sum.plottingHelpers.shap_analysis import analyze_best_model_all_folds
from Stress_study_ear_sum.config import Config


def ablationTag():
    if not Config.excludeModalities:
        return 'allModalities'
    return 'no_' + '_'.join(sorted(m.upper() for m in Config.excludeModalities))


def main():
    config = Config()
    classNames = list(config.classLabels.keys())
    tag = ablationTag()

    pipeline = MLPipeline()
    X, yClass, yReg, subjects, sessions, featureNames = pipeline.loadAndProcess()

    for splitMethod in config.splitMethod:
        print(f"Split method {splitMethod.upper()} features {tag}")
        splits = pipeline.getSplits(yClass, subjects, sessions, splitMethod)

        # classification does not depend on the regression target, so run it once
        clsResults, clsPreds, clsModels = pipeline.evaluateClassifiers(
            X, yClass, subjects, splits, splitMethod)
        bestCls, bestF1 = pipeline.findBestModel(clsResults, 'f1_macro')
        clsYTrue = {name: p['y_true'] for name, p in clsPreds.items()}
        clsYPred = {name: p['y_pred'] for name, p in clsPreds.items()}

        for targetIdx in config.regTargets:
            targetName = config.regTargetNames[targetIdx]
            methodOutputDir = os.path.join(config.outputDir, tag, splitMethod, targetName)
            methodShapDir = os.path.join(config.shapDir, tag, splitMethod, targetName)
            os.makedirs(methodOutputDir, exist_ok=True)
            regResults, regPreds, regModels = pipeline.evaluateRegressors(
                X, yClass, yReg, subjects, sessions, splits, splitMethod, targetIdx=targetIdx)
            bestReg, bestR2 = pipeline.findBestModel(regResults, 'r2')

            create_combined_report(classification_results=clsResults, regression_results=regResults, cls_y_true_dict=clsYTrue, cls_y_pred_dict=clsYPred,
                                   reg_y_true_dict={name: np.array(p['y_true']) for name, p in regPreds.items()}, reg_y_pred_dict={name: np.array(p['y_pred']) for name, p in regPreds.items()},
                                   class_names=classNames, output_dir=methodOutputDir, target_name=targetName)

            if config.runShap:
                analyze_best_model_all_folds(fold_data_list=clsModels[bestCls], model_name=f"{bestCls}_classification", feature_names=featureNames,
                                             class_names=classNames, output_dir=os.path.join(methodShapDir, 'classification'))
                analyze_best_model_all_folds(fold_data_list=regModels[bestReg], model_name=f"{bestReg}_regression_{targetName}",
                                             feature_names=featureNames, class_names=None, output_dir=os.path.join(methodShapDir, 'regression'))


if __name__ == '__main__':
    main()
