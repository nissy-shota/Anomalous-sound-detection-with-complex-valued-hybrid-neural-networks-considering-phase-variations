import os
import sys
from typing import Dict, List, Tuple

import joblib
import numpy as np
import numpy.typing as npt
import scipy.stats
import torch
from omegaconf import DictConfig
from scipy.special import softmax
from sklearn import metrics, preprocessing

from preprocessing.feature_extraction import LoadMultiLabelFeatures
from utils import util_dirs, util_files


def calc_anomaly_score(
    model,
    dir_path: str,
    machine_classifier_index: np.int64,
    config: DictConfig,
    device,
):
    """
    Calculate anomaly score.
    """
    dim_fft = int(config.preprocessing.feature.n_fft / 2 + 1)
    transform = LoadMultiLabelFeatures(is_eval=False, cfg=config)
    try:
        # extract features (log-mel spectrogram)
        data = transform(dir_path)
    except FileNotFoundError:
        print("File broken!!: {}".format(dir_path))

    condition = np.zeros((data.shape[0]), dtype=int)
    if machine_classifier_index != -1:
        condition[:] = machine_classifier_index

    feed_data = data.to(device).type(torch.complex64)
    feed_data = torch.unsqueeze(feed_data, dim=0)
    with torch.no_grad():
        output = model(feed_data)  # notice: unnormalized output
        output = output.to("cpu").detach().numpy().copy()  # tensor to np array.

    output = np.exp(output)
    prob = output[:, machine_classifier_index]

    y_pred = np.mean(
        np.log(
            np.maximum(1.0 - prob, sys.float_info.epsilon)
            - np.log(np.maximum(prob, sys.float_info.epsilon))
        )
    )

    return y_pred


def inference_calc_anomaly_score(
    model,
    dir_path: str,
    machine_classifier_index: np.int64,
    config: DictConfig,
    device,
):
    """
    Calculate anomaly score.
    """
    dim_fft = int(config.preprocessing.feature.n_fft / 2 + 1)
    transform = LoadMultiLabelFeatures(is_eval=True, cfg=config)
    try:
        # extract features (log-mel spectrogram)
        data = transform(dir_path)
    except FileNotFoundError:
        print("File broken!!: {}".format(dir_path))

    condition = np.zeros((data.shape[0]), dtype=int)
    if machine_classifier_index != -1:
        condition[:] = machine_classifier_index

    # feed_data = torch.from_numpy(data).clone()
    feed_data = data.to(device).type(torch.complex64)
    splited_feed_data = torch.tensor_split(feed_data, config.test.inference_split)

    all_output = np.empty((feed_data.shape[0], model.classifier.out_features))
    with torch.no_grad():
        start_index = 0
        for data in splited_feed_data:
            output = model(data)  # notice: unnormalized output
            output = output.to("cpu").detach().numpy().copy()  # tensor to np array.
            end_index = start_index + output.shape[0]
            all_output[start_index:end_index] = output
            start_index = end_index

    all_output = np.exp(all_output)
    prob = all_output[:, machine_classifier_index]

    y_pred = np.mean(
        np.log(
            np.maximum(1.0 - prob, sys.float_info.epsilon)
            - np.log(np.maximum(prob, sys.float_info.epsilon))
        )
    )

    return y_pred


def fit_gaussian_dist(
    model, target_dir, mode: str, config: DictConfig, device
) -> Tuple[float, float]:

    unique_case_ids = np.unique(
        util_files.get_machine_case_id(target_dir=target_dir, mode=mode)
    )
    dataset_scores = np.array([], dtype=np.float64)

    # calculate anomaly scores over machine_ids
    for machine_id, machine_id_name in enumerate(unique_case_ids):
        _dirs = util_dirs.generate_dir_list(target_dir, machine_id_name, mode=mode)
        machine_id_scores = [0.0 for k in _dirs]
        for dir_idx, dir_path in enumerate(_dirs):
            machine_id_scores[dir_idx] = inference_calc_anomaly_score(
                model,
                dir_path=dir_path,
                config=config,
                machine_classifier_index=machine_id,
                device=device,
            )

        machine_id_scores = np.array(machine_id_scores)
        dataset_scores = np.append(dataset_scores, machine_id_scores)

    dataset_scores = np.array(dataset_scores)

    gaussian_params = scipy.stats.norm.fit(dataset_scores)
    gaussian_params = list(gaussian_params)

    score_file_path = "{model}/score_distr_{machine_type}.pkl".format(
        model=config.data_structures.model_directory,
        machine_type=os.path.split(target_dir)[1],
    )

    joblib.dump(gaussian_params, score_file_path)
    loc = gaussian_params[0]
    scale = gaussian_params[1]

    return loc, scale


def calc_decision_threshold(target_dir: str, config: DictConfig) -> np.float64:
    """
    Calculate decision_threshold from anomaly score distribution.
    """

    # load anomaly score distribution for determining threshold
    score_distr_file_path = "{model}/score_distr_{machine_type}.pkl".format(
        model=config.data_structures.model_directory,
        machine_type=os.path.split(target_dir)[1],
    )

    if config.metrics.metric_pdf == "gaussian":

        loc_hat, scale_hat = joblib.load(score_distr_file_path)
        decision_threshold = scipy.stats.norm.ppf(
            q=config.metrics.decision_threshold,
            loc=loc_hat,
            scale=scale_hat,
        )

    return decision_threshold


def save_anomaly_score(
    score_list: Dict,
    target_dir: str,
    machine_id: str,
    dir_name: str,
    config: DictConfig,
):
    """
    Save anomaly scores and decision results.

    score_list : anomaly scores and decision results (type: dictionary).
    """

    # output anomaly scores
    util_files.save_csv(
        save_file_path="{result}/anomaly_score_{machine_type}"
        "_{machine_id}_{dir_name}.csv".format(
            result=config.data_structures.result_directory,
            machine_type=os.path.split(target_dir)[1],
            machine_id=machine_id,
            dir_name=dir_name,
        ),
        save_data=score_list["anomaly"],
    )

    # output decision results
    util_files.save_csv(
        save_file_path="{result}/decision_result_{machine_type}"
        "_{machine_id}_{dir_name}.csv".format(
            result=config.data_structures.result_directory,
            machine_type=os.path.split(target_dir)[1],
            machine_id=machine_id,
            dir_name=dir_name,
        ),
        save_data=score_list["decision"],
    )


def calc_evaluation_scores(
    y_true: npt.NDArray[np.float64],
    y_pred: List[np.float32],
    decision_threshold: np.float64,
    config: DictConfig,
) -> Tuple[np.float64, np.float64, np.float64, np.float64, np.float64]:
    """
    Calculate evaluation scores (AUC, pAUC, precision, recall, and F1 score)
    """
    auc = metrics.roc_auc_score(y_true, y_pred)
    p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=config.metrics.max_fpr)

    (_, false_positive, false_negative, true_positive,) = metrics.confusion_matrix(
        y_true, [1 if x > decision_threshold else 0 for x in y_pred]
    ).ravel()

    prec = true_positive / np.maximum(
        true_positive + false_positive, sys.float_info.epsilon
    )
    recall = true_positive / np.maximum(
        true_positive + false_negative, sys.float_info.epsilon
    )
    f1_score = 2.0 * prec * recall / np.maximum(prec + recall, sys.float_info.epsilon)

    print("AUC : {:.6f}".format(auc))
    print("pAUC : {:.6f}".format(p_auc))
    print("precision : {:.6f}".format(prec))
    print("recall : {:.6f}".format(recall))
    print("F1 score : {:.6f}".format(f1_score))

    return auc, p_auc, prec, recall, f1_score
