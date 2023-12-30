import argparse
import csv
import glob
import itertools
import os
import re
import sys
from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt
from itertools import chain


def generate_dir_list(target_dir: str, unique_case_id: str, mode: str) -> List[str]:

    query = f"{target_dir}/{mode}/{unique_case_id}/*"
    dirs = glob.glob(query)

    return dirs


def generate_dir_list_and_label(
    target_dir: str, case_id: str, mode: str
) -> Tuple[npt.NDArray[np.str], npt.NDArray[np.float64]]:

    normal_dir_name = "NormalSound_IND"
    anomaly_dir_name = "AnomalousSound_IND"

    query = f"{target_dir}/{mode}/{case_id}/{normal_dir_name}/*"
    normal_dirs = sorted(glob.glob(query))
    normal_labels = np.zeros(len(normal_dirs))

    query = f"{target_dir}/{mode}/{case_id}/{anomaly_dir_name}/*"

    anomaly_each_ab_num_dirs = sorted((glob.glob(query)))

    temp_anomaly_dirs = []
    for anomaly_each_ab_num_dir in anomaly_each_ab_num_dirs:
        query = f"{anomaly_each_ab_num_dir}/*"
        temp_anomaly_dirs.append(sorted(glob.glob(query)))

    anomaly_dirs = list(chain.from_iterable(temp_anomaly_dirs))
    del temp_anomaly_dirs

    anomaly_labels = np.ones(len(anomaly_dirs))

    dirs = np.concatenate((normal_dirs, anomaly_dirs), axis=0)
    labels = np.concatenate((normal_labels, anomaly_labels), axis=0)

    assert len(dirs) == len(labels)

    return dirs, labels
