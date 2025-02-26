from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Any
import pandas as pd
class BaseTrainer(ABC):
    """General Interface for all trainers"""

    @abstractmethod
    def load(self, model_path: Path, task: str) -> Any:
        """load model method implemented by all trainers"""
        pass

    @abstractmethod
    def preprocess(self, data_path: Path, task: str) -> Any:
        """preprocess data method based on frameworks implemented by all trainers"""
        pass


    @abstractmethod
    def train(self, train_data: Path, val_data: Optional[Path], model_path: Path) -> Any:
        """training method implemented by all trainers"""
        pass

    @abstractmethod
    def predict(self, model_path: Path, sample_data_path: Path, training_infor) -> Any:
        """predict method implemented by all trainers"""
        pass

    @abstractmethod
    def evaluate(self, model_path: Path, test_data: pd.DataFrame, label: str, task: str) -> dict:
        """evaluate method implemented by all trainers"""
        pass

    @abstractmethod
    def export(self, saved_model_path: Path, saved_metadata_path: Path) -> dict:
        """export model and save metadata method implemented by all trainers"""
        pass
    