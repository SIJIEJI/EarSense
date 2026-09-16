"""Offline analysis of three-channel EarSense PPG recordings."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ppg_io import iter_ppg_windows
from ppg_output import save_analysis_results, save_filtered_signals


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze green, infrared, and red PPG channels in an Excel file."
    )
    parser.add_argument("input", type=Path, help="Input .xlsx file")
    parser.add_argument("--output-dir", type=Path, help="Output directory (default: beside input)")
    parser.add_argument("--sampling-rate", type=float, default=95.0, help="PPG sampling rate in Hz")
    parser.add_argument("--window-seconds", type=float, default=30.0, help="Analysis window length")
    parser.add_argument("--overlap", type=float, default=0.0, help="Window overlap as a fraction [0, 1)")
    return parser


def analyze_file(input_path: Path, output_dir: Path, sampling_rate: float,
                 window_seconds: float, overlap: float) -> tuple[Path, Path]:
    from helperFiles.ppgAnalysisHelpers import ppgAnalysisProtocol

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")
    if input_path.suffix.lower() != ".xlsx":
        raise ValueError("Input must be an .xlsx file")
    if sampling_rate <= 0 or window_seconds <= 0:
        raise ValueError("Sampling rate and window length must be positive")
    if not 0 <= overlap < 1:
        raise ValueError("Overlap must satisfy 0 <= overlap < 1")

    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = output_dir / f"{input_path.stem}_ppg_analysis.xlsx"
    filtered_path = output_dir / f"{input_path.stem}_ppg_filtered.xlsx"

    analyzer = ppgAnalysisProtocol(
        [0.3, 1.2], [0.5, 5.0], [0.5, 3.0], [0.5, 3.0],
        [0.1, 0.5], 0.3, sampling_rate,
    )

    metrics: list[tuple[float, ...]] = []
    hrv: list[tuple[float, float]] = []
    times: list[float] = []
    filtered: list[tuple[np.ndarray, ...]] = []

    for window in iter_ppg_windows(input_path, window_seconds, sampling_rate, overlap):
        time, green, infrared, red = window.T
        result = analyzer.analyze(green, infrared, red)
        heart_rate, breathing_rate, spo2, ratio = result[:4]
        green_hr, green_br, ir_filtered, red_filtered = result[4:]
        breath_metrics = analyzer.compute_breath_amplitude(green_br, plot=False)

        metrics.append((heart_rate, breathing_rate, spo2, ratio, *breath_metrics))
        hrv.append(analyzer.compute_hrv_rmssd_sdnn(green_hr))
        times.append(float(time[len(time) // 2]))
        filtered.append((time, -green_hr, -green_br, ir_filtered, red_filtered))

    if not metrics:
        raise ValueError("No complete PPG analysis window was found in the input file")

    save_analysis_results(analysis_path, times, metrics, hrv)
    save_filtered_signals(filtered_path, filtered, sampling_rate)
    return analysis_path, filtered_path


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir or args.input.resolve().parent
    analysis_path, filtered_path = analyze_file(
        args.input.resolve(), output_dir.resolve(), args.sampling_rate,
        args.window_seconds, args.overlap,
    )
    print(f"Saved metrics: {analysis_path}")
    print(f"Saved filtered signals: {filtered_path}")


if __name__ == "__main__":
    main()
