#!/bin/bash

# Large
poetry run python training.py preprocessing=complex_spectrogram.yaml dataloader=default.yaml experiments.MLFLOW_RUN_NOTE="Large complex_spectrogram"
poetry run python test.py preprocessing=complex_spectrogram.yaml experiments.MLFLOW_RUN_NOTE="Large complex_spectrogram"

poetry run python training.py preprocessing=log_complex_spectrogram.yaml dataloader=default.yaml experiments.MLFLOW_RUN_NOTE="Large log_complex_spectrogram"
poetry run python test.py preprocessing=log_complex_spectrogram.yaml experiments.MLFLOW_RUN_NOTE="Large log_complex_spectrogram"

poetry run python training.py preprocessing=log_and_raw_complex_spectrogram.yaml dataloader=default.yaml experiments.MLFLOW_RUN_NOTE="Large log_and_raw_complex_spectrogram"
poetry run python test.py preprocessing=log_and_raw_complex_spectrogram.yaml training.batch_size=32 experiments.MLFLOW_RUN_NOTE="Large log_and_raw_complex_spectrogram"


# Small
poetry run python training.py preprocessing=complex_spectrogram.yaml dataloader=small1000.yaml experiments.MLFLOW_RUN_NOTE="small1000 complex_spectrogram"
poetry run python test.py preprocessing=complex_spectrogram.yaml experiments.MLFLOW_RUN_NOTE="small1000 complex_spectrogram"

poetry run python training.py preprocessing=log_complex_spectrogram.yaml dataloader=small1000.yaml experiments.MLFLOW_RUN_NOTE="small1000 log_complex_spectrogram"
poetry run python test.py preprocessing=log_complex_spectrogram.yaml experiments.MLFLOW_RUN_NOTE="small1000 log_complex_spectrogram"

poetry run python training.py preprocessing=log_and_raw_complex_spectrogram.yaml dataloader=small1000.yaml experiments.MLFLOW_RUN_NOTE="small1000 log_and_raw_complex_spectrogram"
poetry run python test.py preprocessing=log_and_raw_complex_spectrogram.yaml training.batch_size=32 experiments.MLFLOW_RUN_NOTE="small1000 log_and_raw_complex_spectrogram"