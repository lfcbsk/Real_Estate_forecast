import pickle

import numpy as np
import onnxruntime as ort

from src.models.model_config import ModelConfig


class ModelRegistry:
    def __init__(self):
        self.session = ort.InferenceSession(str(ModelConfig.MODEL_PATH))

        self.input_name = self.session.get_inputs()[0].name

        self.features = self._load_pickle(ModelConfig.FEATURE_LIST_PATH)

        self.sector_stats = self._load_pickle(ModelConfig.SECTOR_STATS_PATH)

        self.sector_profile = self._load_pickle(ModelConfig.SECTOR_PROFILE_PATH)

        self.zero_sectors = self._load_pickle(ModelConfig.ZERO_SECTOR_PATH)

    @staticmethod
    def _load_pickle(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def predict(self, X):
        X = X[self.features].fillna(0).astype(np.float32)

        pred = self.session.run(None, {self.input_name: X.values})[0]

        return pred
