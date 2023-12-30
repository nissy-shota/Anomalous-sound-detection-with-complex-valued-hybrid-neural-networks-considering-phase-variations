import os

from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import (
    MLFLOW_GIT_REPO_URL,
    MLFLOW_RUN_NAME,
    MLFLOW_SOURCE_NAME,
    MLFLOW_USER,
    MLFLOW_GIT_COMMIT,
    MLFLOW_RUN_NOTE,
)
from omegaconf import DictConfig, ListConfig


class MlflowWriter:
    def __init__(self, experiment_name, cfg, script_file_name):

        artifacts_dir = os.path.abspath(cfg.data_structures.artifacts_directory)
        self.tracking_uri = os.path.join(artifacts_dir, "mlruns")
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

        try:
            self.experiment_id = self.client.create_experiment(experiment_name)
        except Exception as e:
            print(e)
            self.experiment_id = self.client.get_experiment_by_name(
                experiment_name
            ).experiment_id

        self.tags = {
            MLFLOW_RUN_NAME: cfg.experiments.MLFLOW_RUN_NAME,
            MLFLOW_USER: cfg.experiments.MLFLOW_USER,
            MLFLOW_SOURCE_NAME: script_file_name,
            MLFLOW_GIT_REPO_URL: cfg.experiments.MLFLOW_GIT_REPO_URL,
            MLFLOW_GIT_COMMIT: cfg.experiments.MLFLOW_GIT_COMMIT,
            MLFLOW_RUN_NOTE: cfg.experiments.MLFLOW_RUN_NOTE,
        }

        self.experiment = self.client.get_experiment(self.experiment_id)
        print("New experiment started")
        print(f"Experiment Name: {self.experiment.name}")
        print(f"Experiment id: {self.experiment.experiment_id}")
        print(f"Artifact Location: {self.experiment.artifact_location}")

    def log_params_from_omegaconf_dict(self, params):
        for param_name, element in params.items():
            self._explore_recursive(param_name, element)

    def _explore_recursive(self, parent_name, element):
        if isinstance(element, DictConfig):
            for k, v in element.items():
                if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                    self._explore_recursive(f"{parent_name}.{k}", v)
                else:
                    self.client.log_param(self.run_id, f"{parent_name}.{k}", v)
        elif isinstance(element, ListConfig):
            for i, v in enumerate(element):
                self.client.log_param(self.run_id, f"{parent_name}.{i}", v)
        else:
            self.client.log_param(self.run_id, f"{parent_name}", element)

    def log_param(self, key, value):
        self.client.log_param(self.run_id, key, value)

    def log_metric(self, key, value):
        self.client.log_metric(self.run_id, key, value)

    def log_metric_step(self, key, value, step):
        self.client.log_metric(self.run_id, key, value, step=step)

    def log_artifact(self, local_path):
        self.client.log_artifact(self.run_id, local_path)

    def log_artifacts(self, local_path):
        self.client.log_artifacts(self.run_id, local_path)

    def log_dict(self, dictionary, file):
        self.client.log_dict(self.run_id, dictionary, file)

    def log_figure(self, figure, file):
        self.client.log_figure(self.run_id, figure, file)

    def set_terminated(self):
        self.client.set_terminated(self.run_id)

    def create_new_run(self, tags=None):
        self.run = self.client.create_run(self.experiment_id, tags=tags)
        self.run_id = self.run.info.run_id
        print(f"New run started: {tags['mlflow.runName']}")
