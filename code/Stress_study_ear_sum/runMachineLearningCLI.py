"""
Run from the repository root
example:
python -m Stress_study_ear_sum.runMachineLearningCLI
python -m Stress_study_ear_sum.runMachineLearningCLI --split rolling_origin --target PA NA --no-shap
python -m Stress_study_ear_sum.runMachineLearningCLI --exclude ACC --output-dir ./results_noACC
python -m Stress_study_ear_sum.runMachineLearningCLI --dry-run   # print the effective Config only
"""
import argparse
from Stress_study_ear_sum.config import Config
from Stress_study_ear_sum.runMachineLearning import main as runPipeline, ablationTag

SPLIT_METHODS = ['rolling_origin', 'loso', 'sgkf', 'skf']
MODALITIES = [m.upper() for m in Config.modalityPrefixes]


def buildParser():
    parser = argparse.ArgumentParser(
        description='Run the stress-study ML pipeline (classification + regression + SHAP) with Config values overridden from the command line.', formatter_class=argparse.RawTextHelpFormatter)

    data = parser.add_argument_group('data / features')
    data.add_argument('--exclude', nargs='*', type=str.upper, choices=MODALITIES, metavar='MODALITY',
                      help=f'feature modalities to drop {MODALITIES}; pass the flag with no value to keep all\n'
                           f'(config.py: {Config.excludeModalities or "none"})')
    data.add_argument('--baseline-classes', nargs='+', type=int, metavar='CLASS',
                      help=f'class ids used as the per-subject normalisation reference {Config.classLabels}\n'
                           f'(config.py: {Config.baselineClasses})')

    task = parser.add_argument_group('task')
    task.add_argument('--split', nargs='+', choices=SPLIT_METHODS, metavar='METHOD',
                      help=f'split method(s) to evaluate {SPLIT_METHODS}\n(config.py: {Config.splitMethod})')
    task.add_argument('--target', nargs='+', type=str.upper, choices=Config.regTargetNames, metavar='TARGET',
                      help=f'regression target(s) {Config.regTargetNames}\n'
                           f'(config.py: {[Config.regTargetNames[i] for i in Config.regTargets]})')
    task.add_argument('--rolling-folds', type=int, metavar='N',
                      help=f'number of rolling-origin folds (config.py: {Config.rollingFolds})')
    task.add_argument('--embargo', type=int, metavar='N',
                      help=f'windows dropped from train at each train/test boundary (config.py: {Config.embargo})')
    task.add_argument('--smooth-window', type=int, metavar='N',
                      help=f'trailing-mean window on regression predictions, <=1 disables (config.py: {Config.smoothWindow})')
    task.add_argument('--center-target', action=argparse.BooleanOptionalAction, default=None,
                      help=f'regress the deviation from each subject\'s baseline score (config.py: {Config.centerRegressionTarget})')

    output = parser.add_argument_group('output')
    output.add_argument('--shap', action=argparse.BooleanOptionalAction, default=None,
                        help=f'run SHAP on the best classifier and regressor (config.py: {Config.runShap})')
    output.add_argument('--output-dir', metavar='DIR', help=f'results root (config.py: {Config.outputDir})')
    output.add_argument('--shap-dir', metavar='DIR', help=f'SHAP results root (config.py: {Config.shapDir})')
    output.add_argument('--dry-run', action='store_true',
                        help='print the effective Config and exit without loading data or training')
    return parser


def applyOverrides(args, parser):
    validClasses = set(Config.classLabels.values())
    if args.baseline_classes is not None and not set(args.baseline_classes) <= validClasses:
        parser.error(f'--baseline-classes must be a subset of {sorted(validClasses)}')
    overrides = {
        'excludeModalities': args.exclude,
        'baselineClasses': args.baseline_classes,
        'splitMethod': args.split,
        'regTargets': None if args.target is None else [Config.regTargetNames.index(t) for t in args.target],
        'rollingFolds': args.rolling_folds,
        'embargo': args.embargo,
        'smoothWindow': args.smooth_window,
        'centerRegressionTarget': args.center_target,
        'runShap': args.shap,
        'outputDir': args.output_dir,
        'shapDir': args.shap_dir,
    }
    for name, value in overrides.items():
        if value is not None:
            setattr(Config, name, value)


def printEffectiveConfig():
    rows = [
        ('feature set', ablationTag()),
        ('excludeModalities', Config.excludeModalities),
        ('baselineClasses', Config.baselineClasses),
        ('splitMethod', Config.splitMethod),
        ('regTargets', [Config.regTargetNames[i] for i in Config.regTargets]),
        ('rollingFolds', Config.rollingFolds),
        ('embargo', Config.embargo),
        ('smoothWindow', Config.smoothWindow),
        ('centerRegressionTarget', Config.centerRegressionTarget),
        ('runShap', Config.runShap),
        ('outputDir', Config.outputDir),
        ('shapDir', Config.shapDir),
    ]
    print('Effective Config')
    for name, value in rows:
        print(f'  {name:<24}{value}')
    print()


def main(argv=None):
    parser = buildParser()
    args = parser.parse_args(argv)
    applyOverrides(args, parser)
    printEffectiveConfig()
    if args.dry_run:
        return
    runPipeline()


if __name__ == '__main__':
    main()
