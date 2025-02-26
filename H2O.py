import gc
from typing import Union
import glob
import shutil
from dataclasses import dataclass
import subprocess
import torchvision.models as models
import torchvision.transforms as transforms
import time
import argparse
import pandas as pd
import logging
import h2o
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from h2o.automl import H2OAutoML
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef,
)
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import asyncio
from typing import Optional, Union
import torch
import os
import json
from base_trainer import BaseTrainer
from base import TrainingInfo
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
torch.cuda.empty_cache()
gc.collect()

torch.set_float32_matmul_precision("medium")

class H2OTrainer(BaseTrainer):
    def __init__(self, kwargs: Optional[dict] = None):
        self.fit_args = None
        self.model_args = None
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)

        self.parse_args(kwargs)
        self.model = None
        self.model_extract_features = None
        self.text_vectorizer = None
        h2o.init(port=12345)

    def parse_args(self, kwargs: Optional[dict] = None):
        if kwargs is None:
            return
        self.model_args = kwargs.setdefault("ag_model_args", {})
        self.fit_args = kwargs.setdefault(
            "ag_fit_args", {}
        )


    def extract_features(self, image_path):
        model = None
        if self.model_extract_features is None:
            model = models.resnet50(pretrained=True)
            model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove the last classification layer
            model.eval()
            self.model_extract_features = model
        else:
            model = self.model_extract_features
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            features = model(image).squeeze().numpy()  # Output shape (2048,)
        
        return features

    def get_features_from_images(self, folder_image_path: str):
        data = []
        lst_folder_images_object = os.listdir(folder_image_path)
        print('List folder images object: ',lst_folder_images_object)
        for folder in lst_folder_images_object:
            for file in glob.glob(os.path.join(folder_image_path, folder, "*.jpg")):
                features = self.extract_features(file)
                features = np.append(features, folder)
                data.append(features)

        df = pd.DataFrame(data)
        print('Data frame features: ',df)   
        df.to_csv(f'{folder_image_path}/features.csv', index=False)
        return df

    def load(self,model_path: Path, task: str):
        try:
            model = h2o.load_model(model_path)
            self.model = model
            return model
        except Exception as e:
            import traceback
            traceback.print_exc()   
            print(e)
            return None 
    
    def preprocess(self, data, training_info: TrainingInfo):
        try:

            match(training_info.task):
                case 'IMAGE_CLASSIFICATION':
                    df = []
                    for _,image_path in data.iterrows():
                        print('Image path: ',image_path[training_info.image_column])
                        features_ = self.extract_features(image_path[training_info.image_column])  
                        features_ = np.append(features_, image_path[training_info.label_column])
                        df.append(features_)
                    if training_info.label_column == '':
                        column_names = [f'feature_{i}' for i in range(len(features_) - 1)]
                    else:
                        column_names = [f'feature_{i}' for i in range(len(features_) - 1)] + [training_info.label_column]
                    df = pd.DataFrame(df,columns=column_names)
                    return df
                case 'TABULAR_CLASSIFICATION':
                    return data
                case 'TEXT_CLASSIFICATION':
                    if self.text_vectorizer is None:
                        self.text_vectorizer = TfidfVectorizer(max_features=1000)
                        text_features = self.text_vectorizer.fit_transform(data.drop(columns=[training_info.label_column]).values.ravel()).toarray()
                    else:
                        if training_info.label_column != '':
                            text_features = self.text_vectorizer.transform(data.drop(columns=[training_info.label_column]).values.ravel()).toarray()
                        else:
                            text_features = self.text_vectorizer.transform(data.values.ravel()).toarray()
                    feature_df = pd.DataFrame(text_features)
                    if training_info.label_column != '':
                        feature_df[training_info.label_column] = data[training_info.label_column]
                    return feature_df
                case 'MULTIMODAL_CLASSIFICATION':
                    print(f"H2O currently does not support this {training_info.task} task")
                    return None
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._logger.error(f"An unexpected error occurred during preprocessing: {e}")
            return None

    def train(
        self,
        training_info: TrainingInfo,
    ):
        try:
            # TEMPORARY
            user_model_path = "/home/quoc/works/Learn/NCKH/model"

            
            train_data = pd.read_csv("/home/quoc/works/Learn/NCKH/loan-approval-tabular-classification/train11.csv")
            test_data = pd.read_csv("/home/quoc/works/Learn/NCKH/loan-approval-tabular-classification/test11.csv")

            train_data = self.preprocess(train_data, training_info)
            test_data = self.preprocess(test_data, training_info) 
            print('-'*10)
            print('Start to train: ---->')
            print('-'*10)
            match(training_info.task):
                case 'IMAGE_CLASSIFICATION':
                    train_df = h2o.H2OFrame(train_data)
                    x = [col for col in train_df.columns if col != training_info.label_column]
                case 'TABULAR_CLASSIFICATION':
                    train_df = h2o.H2OFrame(train_data)
                    categorical_columns = [col for col in train_df.columns if train_df[col].isfactor()[0]]
                    for col in categorical_columns:
                        train_df[col] = train_df[col].asfactor()

                    x = [col for col in train_df.columns if col != training_info.label_column]
                case 'TEXT_CLASSIFICATION':
                    train_df = h2o.H2OFrame(train_data)
                    x = [col for col in train_df.columns if col != training_info.label_column]
                case _:
                    print(f"H2O currently does not support this {training_info.task} task")
                    return None

            self.model = H2OAutoML(max_models=10, seed=42, max_runtime_secs=training_info.training_time)
            self.model.train(x=x, y=training_info.label_column, training_frame=train_df)
            print('Train Success')
            model_path = self.export(f"{user_model_path}", f"{user_model_path}/metadata.json", training_info)
            metrics = self.evaluate(test_data, training_info)
            infor_model = {'model_path': model_path, 'metrics': metrics}
            print('infor_model: ',infor_model)
            h2o.shutdown()
            return None
        except Exception as e:  
            import traceback
            traceback.print_exc()

            self._logger.error(f"An error occurred during training: {e}")
            return None
    
    def evaluate(self, test_data, training_info)  -> dict:
        try:
            df = test_data
            df_features = df.drop(columns=[training_info.label_column])
            y_true = df[training_info.label_column]
            if y_true.dtype == object:
                le = LabelEncoder()
                y_true_encoded = le.fit_transform(y_true)
            else:
                y_true_encoded = y_true.astype(int)  # Nếu đã là số thì giữ nguyên
            
            test_df_h2o = h2o.H2OFrame(df_features)
            y_pred = self.model.predict(test_df_h2o)

            y_pred_df = y_pred.as_data_frame()  
            y_pred_series = y_pred_df.iloc[:, 0] 
           
            if y_pred_series.dtype == float:  # Nếu là xác suất
                y_pred_encoded = (y_pred_series >= 0.5).astype(int)
            elif y_pred_series.dtype == object:  # Nếu là chuỗi (nhãn phân loại)
                y_pred_encoded = le.transform(y_pred_series)  # Sử dụng cùng LabelEncoder
            else:
                y_pred_encoded = y_pred_series.astype(int)  # Nếu đã là số thì giữ nguyên


     
            test_res = {
                "accuracy": accuracy_score(y_true_encoded, y_pred_encoded),
                "balanced_accuracy": balanced_accuracy_score(y_true_encoded, y_pred_encoded),
                "mcc": matthews_corrcoef(y_true_encoded, y_pred_encoded),
                "f1_score": f1_score(y_true_encoded, y_pred_encoded, average="binary"),
                "precision": precision_score(y_true_encoded, y_pred_encoded, average="binary"),
                "recall": recall_score(y_true_encoded, y_pred_encoded, average="binary"),
            }
            print('Test result: ',test_res)
            return test_res

        except Exception as e:
            import traceback
            traceback.print_exc()
            self._logger.error(f"An unexpected error occurred during evaluation: {e}")
            return None

    def predict(self, model_path, sample_data_path, training_info):
        try:
            model = h2o.load_model(str(model_path))   # Lấy model tốt nhất
            df = pd.read_csv(sample_data_path)
            if training_info.task == 'IMAGE_CLASSIFICATION':
                df = self.preprocess(df, training_info)
                y_pred = model.predict(h2o.H2OFrame(df))
                y_pred_df = y_pred.as_data_frame()
                print('Predict result: ',y_pred_df)
            elif training_info.task == 'TEXT_CLASSIFICATION':
                last_path = str(model_path).split('/')[-1]
                vectorize_path = str(model_path).replace(last_path, 'text_vectorize.pkl')
                self.text_vectorizer = joblib.load(vectorize_path)
                df = self.preprocess(df, training_info)
                y_pred = model.predict(h2o.H2OFrame(df))
                y_pred_df = y_pred.as_data_frame()
                print('Predict result: ',y_pred_df)

            y_pred_series = y_pred_df.iloc[:, 0] 
           
            if y_pred_series.dtype == float:  # Nếu là xác suất
                y_pred_encoded = (y_pred_series >= 0.5).astype(int)
            print(y_pred_encoded)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._logger.error(f"An unexpected error occurred during prediction: {e}")
            return None
    
    def postprocess(self, data, training_info: TrainingInfo):
        pass

    def export(self, saved_model_path: str, saved_metadata_path: str, training_info):
        try:
            if self.text_vectorizer:
                joblib.dump(self.text_vectorizer, f"{saved_model_path}/text_vectorize.pkl")
            model_path = h2o.save_model(self.model.leader, path=saved_model_path, force=True, filename=f"{training_info.task_id}_h2o_predictor")
            model = self.model.leader
            model_metadata = {
            "model_id": model.model_id,
            "algo": model.algo,
            "training_time": model.run_time,
            "parameters": model.params,  # Toàn bộ tham số
            "train_rmse": model.rmse(train=True),
            "valid_rmse": model.rmse(valid=True) if model._model_json["output"]["validation_metrics"] else None,
            "valid_auc": model.auc(valid=True) if hasattr(model, "auc") else None,
        }
            with open(saved_metadata_path, "w") as f:
                json.dump(model_metadata, f, indent=4)
            return model_path
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._logger.error(f"An unexpected error occurred during export: {e}")
            return None

    # TODO: what is this function for?
    def save_pipeline(self, pipeline_path):
        # This is func for TPOT, not important
        return None
    
if __name__ == "__main__":
    print('Da vao day')
    trainer = H2OTrainer()
    print('Da vao day')
    trainer.train(
        TrainingInfo(
            task="TABULAR_CLASSIFICATION",
            task_id="1",
            framework="H2O",
            training_time=60*60,
            presets="medium_quality",
            text_column="sentence",
            label_column="loan_status",
            image_column="image",
        )
    )
    # trainer.predict(
    #     Path("/drive2/quocnda/Test_Framework/model/1_h2o_predictor"),
    #     Path("/drive2/quocnda/Test_Framework/dataset_text/test.csv"),
    #     TrainingInfo(
    #         task="TEXT_CLASSIFICATION",
    #         task_id="1",
    #         framework="H2O",
    #         training_time=60,
    #         presets="medium_quality",
    #         text_column="sentence",
    #         label_column='',
    #         image_column="image",
    #     )
    # )


# import pandas as pd
# data = pd.read_csv("/home/quoc/works/Learn/NCKH/loan-approval-tabular-classification/test.csv")
# data = data.iloc[:100]
# data.to_csv("/home/quoc/works/Learn/NCKH/loan-approval-tabular-classification/test11.csv",index=False)