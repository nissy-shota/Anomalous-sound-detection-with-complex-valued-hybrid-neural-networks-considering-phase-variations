import argparse
import csv
import glob
import itertools
import os
import re
import sys
from typing import Dict, List, Tuple

import librosa
import librosa.core
import librosa.feature
import numpy as np
import numpy.typing as npt
import yaml


def get_machine_name_path(dataset_dir: str) -> List[str]:

    print("load_directory [ToyADMOS]")
    query = os.path.abspath("{base}/*".format(base=dataset_dir))
    dirs = sorted(glob.glob(query))
    dirs = [f for f in dirs if os.path.isdir(f)]
    return dirs


def get_machine_case_id(target_dir: str, mode: str) -> List[str]:

    query = os.path.abspath(f"{target_dir}/{mode}/*")
    dirs = sorted(glob.glob(query))
    dirs = [os.path.split(f)[1] for f in dirs if os.path.isdir(f)]
    return dirs


def save_csv(save_file_path, save_data):
    """
    Save results (AUCs and pAUCs) into csv file.
    """
    with open(save_file_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file, lineterminator="\n")
        writer.writerows(save_data)


def save_result(csv_lines, machine_type, config):
    """
    Save averages for AUCs and pAUCs.
    """

    result_path = "{result}/{machine_type}_{file_name}".format(
        result=config.data_structures.result_directory,
        machine_type=machine_type,
        file_name=config.data_structures.result_file,
    )
    print("results -> {}".format(result_path))
    save_csv(save_file_path=result_path, save_data=csv_lines)
