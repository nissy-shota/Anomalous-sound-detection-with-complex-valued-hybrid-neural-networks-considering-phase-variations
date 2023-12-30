#!/bin/bash
poetry run python training.py preprocessing=complex_spectrogram.yaml experiments.MLFLOW_RUN_NOTE="complex spectrogram"
poetry run python test.py preprocessing=complex_spectrogram.yaml

poetry run python training.py preprocessing=log_complex_spectrogram.yaml experiments.MLFLOW_RUN_NOTE="log complex spectrogram"
poetry run python test.py preprocessing=log_complex_spectrogram.yaml

poetry run python training.py preprocessing=log_and_raw_complex_spectrogram.yaml experiments.MLFLOW_RUN_NOTE="complex spectrogram logcomplex spectrogram"
poetry run python test.py preprocessing=log_and_raw_complex_spectrogram.yaml