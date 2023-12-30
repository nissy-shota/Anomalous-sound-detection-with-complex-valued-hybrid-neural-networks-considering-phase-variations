import os
import random
import sys
from typing import Dict, List, Tuple

import hydra
import joblib
import mlflow
import numpy as np
import numpy.typing as npt
import scipy.stats
import torch
from dotenv import load_dotenv
from mlflow.utils.mlflow_tags import (
    MLFLOW_GIT_REPO_URL,
    MLFLOW_RUN_NAME,
    MLFLOW_SOURCE_NAME,
    MLFLOW_USER,
)
from omegaconf import DictConfig

from experiments import mlflow_writer
from metrics import anomaly_socre_metric
from utils import util_dirs, util_files, util_models, util_optimizers

# String constant: "cuda:0" or "cpu"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def torch_fix_seed(seed=44):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


def test_section(
    model,
    test_dirs: List[str],
    decision_threshold: np.float64,
    score_list: Dict[str, None],
    machine_classifier_index: np.int64,
    config: DictConfig,
    device,
) -> List[np.float32]:
    """
    Test a section (almost equal to machine id).
    """
    # section_idx = section_info[1]

    # setup anomaly score file path
    anomaly_score_list = []

    # setup decision result file path
    decision_result_list = []

    y_pred = [0.0 for k in test_dirs]
    for dir_idx, dir_path in enumerate(test_dirs):
        y_pred[dir_idx] = anomaly_socre_metric.inference_calc_anomaly_score(
            model,
            dir_path=dir_path,
            machine_classifier_index=machine_classifier_index,
            config=config,
            device=device,
        )
        anomaly_score_list.append([dir_path, y_pred[dir_idx]])

        # store decision results
        if y_pred[dir_idx] > config.metrics.decision_threshold:
            decision_result_list.append([dir_path, 1])
        else:
            decision_result_list.append([dir_path, 0])

    score_list["anomaly"] = anomaly_score_list
    score_list["decision"] = decision_result_list

    return y_pred


def calc_performance_all(performance, csv_lines):
    """
    Calculate model performance over all sections.
    """
    csv_lines.append(["", "", "AUC", "pAUC", "precision", "recall", "F1 score"])
    amean_performance = np.mean(np.array(performance, dtype=float), axis=0)
    csv_lines.append(
        ["arithmetic mean over all machine types, sections, and domains", ""]
        + list(amean_performance)
    )
    hmean_performance = scipy.stats.hmean(
        np.maximum(np.array(performance, dtype=float), sys.float_info.epsilon),
        axis=0,
    )
    csv_lines.append(
        ["harmonic mean over all machine types, sections, and domains", ""]
        + list(hmean_performance)
    )
    csv_lines.append([])

    return csv_lines


def calc_performance_section(performance, csv_lines):
    """
    Calculate model performance per section.
    """
    amean_performance = np.mean(np.array(performance, dtype=float), axis=0)
    csv_lines.append(["arithmetic mean", ""] + list(amean_performance))
    hmean_performance = scipy.stats.hmean(
        np.maximum(np.array(performance, dtype=float), sys.float_info.epsilon),
        axis=0,
    )
    csv_lines.append(["harmonic mean", ""] + list(hmean_performance))
    csv_lines.append([])

    return csv_lines


@hydra.main(version_base=None, config_path="configures/", config_name="config.yaml")
def main(cfg: DictConfig) -> None:

    torch_fix_seed()
    os.makedirs(cfg.data_structures.result_directory, exist_ok=True)

    dir_list = util_files.get_machine_name_path(
        dataset_dir=cfg.data_structures.dataset_path
    )

    for idx, target_dir in enumerate(dir_list):
        machine_type = os.path.split(target_dir)[1]
        print(f"[{idx+1}/{len(dir_list)}]: {target_dir}")

        if machine_type not in cfg.dataset.target_machine_types:
            print(f"{machine_type} is skipped.")
            continue

        csv_lines = []
        performance = {"section": None, "all": None}

        score_list = {"anomaly": None, "decision": None}

        performance["all"] = []

        EXPERIMENT_NAME = machine_type
        writer = mlflow_writer.MlflowWriter(
            EXPERIMENT_NAME, cfg, cfg.test.script_file_name
        )

        print(f"Features: {cfg.preprocessing.name}")
        print(f"Model: {cfg.models.model_name}")

        tags = {
            MLFLOW_RUN_NAME: cfg.experiments.MLFLOW_RUN_NAME,
            MLFLOW_USER: cfg.experiments.MLFLOW_USER,
            MLFLOW_SOURCE_NAME: "test.py",
            MLFLOW_GIT_REPO_URL: cfg.experiments.MLFLOW_GIT_REPO_URL,
        }
        writer.create_new_run(tags)
        tags["MLFLOW_PARENT_RUN_ID"] = writer.run_id
        writer.log_params_from_omegaconf_dict(cfg)

        machine_case_id_file_path = (
            f"{cfg.data_structures.model_directory}/machine_case_id_{machine_type}.pkl"
        )

        trained_case_id = joblib.load(machine_case_id_file_path)
        n_case_id = trained_case_id.shape[0]

        model = util_models.load_model(
            machine_type=machine_type,
            n_machine_id=n_case_id,
            config=cfg,
            device=DEVICE,
        )

        decision_threshold = anomaly_socre_metric.calc_decision_threshold(
            target_dir, config=cfg
        )
        writer.log_metric("decision threshold", decision_threshold)

        csv_lines.append([os.path.split(target_dir)[1]])  # append machine type
        csv_lines.append(
            ["caseID", "domain", "AUC", "pAUC", "precision", "recall", "F1 score"]
        )
        performance["caseID"] = []

        unique_case_ids = np.unique(
            util_files.get_machine_case_id(target_dir=target_dir, mode="train")
        )

        for case_id in unique_case_ids:

            # search for section_name
            temp_array = np.nonzero(trained_case_id == case_id)[0]
            if temp_array.shape[0] == 0:
                machine_classifier_index = -1
            else:
                machine_classifier_index = temp_array[0]

            test_dirs, y_true = util_dirs.generate_dir_list_and_label(
                target_dir, case_id, mode="test"
            )

            y_pred = test_section(
                model,
                test_dirs,
                decision_threshold,
                score_list,
                machine_classifier_index,
                config=cfg,
                device=DEVICE,
            )

            anomaly_socre_metric.save_anomaly_score(
                score_list=score_list,
                target_dir=target_dir,
                machine_id=case_id,
                dir_name="test",
                config=cfg,
            )

            (
                auc,
                p_auc,
                prec,
                recall,
                f1_score,
            ) = anomaly_socre_metric.calc_evaluation_scores(
                y_true, y_pred, decision_threshold, config=cfg
            )

            eval_scores = (auc, p_auc, prec, recall, f1_score)

            writer.log_metric(case_id + "auc", auc)
            writer.log_metric(case_id + "p_auc", p_auc)
            writer.log_metric(case_id + "precision", prec)
            writer.log_metric(case_id + "recall", recall)
            writer.log_metric(case_id + "f1 score", f1_score)

            csv_lines.append(
                [
                    case_id,
                    "test",
                    *eval_scores,  # unpack
                ]
            )
            performance["caseID"].append(eval_scores)
            performance["all"].append(eval_scores)

        csv_lines = calc_performance_section(performance["caseID"], csv_lines)
        util_files.save_result(csv_lines, machine_type, config=cfg)
        writer.log_artifact(cfg.data_structures.result_directory)
        writer.set_terminated()


if __name__ == "__main__":
    main()
