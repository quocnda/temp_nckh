import gc
import glob
import shutil
import argparse
import time
import json
import logging
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

import joblib
import numpy as np
from pathlib import Path
from tpot import TPOTClassifier, TPOTRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from base_trainer import BaseTrainer
from base import TrainingInfo
from sklearn.decomposition import PCA
gc.collect()

class TPOTTrainer(BaseTrainer):
    def __init__(self, kwargs=None):
        self.model = None
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)
        self.parse_args(kwargs)

        self.image_model = models.resnet50(pretrained=True)
        self.image_model = torch.nn.Sequential(*(list(self.image_model.children())[:-1]))  
        self.image_model.eval()

        self.text_vectorizer = None  

    def parse_args(self, kwargs=None):
        if kwargs is None:
            return
        self.fit_args = kwargs.setdefault("tpot_fit_args", {
            "generations": kwargs.get("generations", 5),
            "population_size": kwargs.get("population_size", 20),
            "cv": kwargs.get("cv", 5),
            "scoring": kwargs.get("scoring", "accuracy"),
            "n_jobs": kwargs.get("n_jobs", -1),
            "verbosity": kwargs.get("verbosity", 2),
            "max_time_mins": kwargs.get("max_time_mins", 3),
        })
        self.label = kwargs.get("label_column", "label")

    def extract_image_features(self, image_path):
        """Trích xuất feature vector từ ảnh."""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            features = self.image_model(image).squeeze().numpy()

        return features

    def process_data(self,df: pd.DataFrame,task: str,label: str):
        # TODO: refactor
        if task == "image-classification":
            feature_list = []
            for _, row in df.iterrows():
                features = self.extract_image_features(row["image"])
                if label:
                    features = np.append(features, row[label])
                feature_list.append(features)
            if label:
                column_names = [f'feature_{i}' for i in range(len(features) - 1)] + [label]
            else:
                column_names = [f'feature_{i}' for i in range(len(features) - 1)]
            feature_df = pd.DataFrame(feature_list,columns=column_names)
            # feature_df[label] = df[label]

        elif task == "text-classification":
            text_features = self.text_vectorizer.fit_transform(df.drop(columns=[label])).toarray()
            feature_df = pd.DataFrame(text_features)
            feature_df[label] = df[label]

        elif task == "multimodal-classification":
            image_features = []
            for _, row in df.iterrows():
                img_feat = self.extract_image_features(row["image"])
                image_features.append(img_feat)

            text_features = self.text_vectorizer.fit_transform(df["text"]).toarray()
            image_df = pd.DataFrame(image_features)
            text_df = pd.DataFrame(text_features)
            other_df = df.drop(columns=["image", "text"])
            feature_df = pd.concat([image_df, text_df, other_df], axis=1)
            feature_df[label] = df[label]

        else:
            feature_df = df.copy()
            categorical_columns = feature_df.select_dtypes(include=["object"]).columns
            label_encoders = {}
            for col in categorical_columns:
                le = LabelEncoder()
                feature_df[col] = le.fit_transform(feature_df[col])
                label_encoders[col] = le
        return feature_df

    def preprocess(self, data, training_info) -> pd.DataFrame:
        """Xử lý dataset thành dạng tabular để TPOT có thể sử dụng."""

        match(training_info.task):
            case "IMAGE_CLASSIFICATION":
                feature_list = []
                for _, row in data.iterrows():
                    features = self.extract_image_features(row["image"])
                    if training_info.label_column:
                        features = np.append(features, row[training_info.label_column])
                    feature_list.append(features)
                # This is for predict which we dont have label columns
                if training_info.label_column: # ????????
                    column_names = [f'feature_{i}' for i in range(len(features) - 1)] + [training_info.label_column]
                else:
                    column_names = [f'feature_{i}' for i in range(len(features) - 1)]
                feature_df = pd.DataFrame(feature_list,columns=column_names)
            case "TEXT_CLASSIFICATION":
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

            case "MULTIMODAL_CLASSIFICATION":
                image_features = []
                for _, row in data.iterrows():
                    img_feat = self.extract_image_features(row[training_info.image_column])
                    image_features.append(img_feat)

                image_df = pd.DataFrame(image_features)
                other_df = data.drop(columns=[training_info.image_column])
                feature_df = pd.concat([image_df, other_df], axis=1)
                feature_df[training_info.label_column] = data[training_info.label_column]
            
            case "TABULAR_CLASSIFICATION":
                feature_df = data.copy()
                categorical_columns = feature_df.select_dtypes(include=["object"]).columns
                label_encoders = {}
                for col in categorical_columns:
                    le = LabelEncoder()
                    feature_df[col] = le.fit_transform(feature_df[col])
                    label_encoders[col] = le
            case _:
                print(f"TPOT not support this {training_info.task} task")
                return None
        return feature_df
        

    def load(self, model_path: Path, task: str):
        model = joblib.load(model_path)
        return model
    
    def export(self, saved_model_path, saved_metadata_path):
        joblib.dump(self.model.fitted_pipeline_, f"{saved_model_path}/tpot_predictor.pkl")
        if self.text_vectorizer:
            joblib.dump(self.text_vectorizer, f"{saved_model_path}/text_vectorize.pkl")
        metadata = {
            "model_name": "TPOT Pipeline",}

        
        with open(saved_metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        return saved_model_path

    def train(self, training_info: TrainingInfo):
        """Huấn luyện TPOT trên dữ liệu đã xử lý."""
        try:
            user_dataset_path = '/home/quoc/works/Learn/NCKH/org-detection-text-classification'
            user_model_path = '/home/quoc/works/Learn/NCKH/model'

            # os.makedirs(user_dataset_path, exist_ok=True)
            # os.makedirs(user_model_path, exist_ok=True)
            
            train_data = pd.read_csv(f"{user_dataset_path}/train.csv")
            test_data = pd.read_csv(f"{user_dataset_path}/test.csv")

            train_data = self.preprocess(train_data, training_info)
            test_data = self.preprocess(test_data, training_info) 
            
            
            
            X = train_data.drop(columns=[training_info.label_column])
            y = train_data[training_info.label_column]

            if training_info.task in ["IMAGE_CLASSIFICATION", "TEXT_CLASSIFICATION", "MULTIMODAL_CLASSIFICATION", "TABULAR_CLASSIFICATION"]:
                model = TPOTClassifier(generations=5, population_size=20, cv=5, n_jobs=-1, verbosity=2, max_time_mins=training_info.training_time // 60)
            else:
                raise ValueError("Invalid task")

            model.fit(X,y)
            self.model = model
            saved_model_path = self.export(f"{user_model_path}", f"{user_model_path}/metadata.json")
            metrics = self.evaluate(test_data, training_info)


            self._logger.info(f"Training completed. Model saved to {saved_model_path}")
            infor_model = {'model_path': saved_model_path, 'metrics': metrics}
            print('Infor model: ',infor_model)
            return None

        except Exception as e:
            import traceback
            traceback.print_exc()
            return None

    def evaluate(self, test_data, training_info):
        """Đánh giá mô hình TPOT."""
        try:
            
            X_test = test_data.drop(columns=[training_info.label_column])
            y_test = test_data[training_info.label_column]
            encoder = LabelEncoder()
            y_test_encoded = encoder.fit_transform(y_test)  # Chuyển nhãn chuỗi thành số

            y_pred = self.model.predict(X_test)
            print('Y_pred: ',y_pred)
            print('Y_test: ',y_test)
            print('Type y_pred: ',type(y_pred))
            print('Type y_test: ',type(y_test))

            # Chuyển y_pred về số theo thứ tự của `encoder`
            y_pred_encoded = encoder.transform(y_pred)  # Chuyển đổi nhãn y_pred về số tương ứng

            y_test = y_test_encoded
            y_pred = y_pred_encoded
            test_res = {
                "accuracy": accuracy_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred, average="weighted"),
                "precision": precision_score(y_test, y_pred, average="weighted"),
                "recall": recall_score(y_test, y_pred, average="weighted"),
                "mcc": matthews_corrcoef(y_test, y_pred),
            }
            print('Test_res: ',test_res)
            return test_res

        except Exception as e:
            import traceback
            traceback.print_exc()
            self._logger.error(f"An unexpected error occurred: {e}")
            return None

    def predict(self,modelpath: Path, sample_data_path: Path,training_info: TrainingInfo):
        try:
            model = joblib.load(modelpath)
            df = pd.read_csv(sample_data_path)
            if training_info.task == 'IMAGE_CLASSIFICATION' :
                feature_df = self.preprocess(df,training_info)
                y_pred = model.predict(feature_df)
            elif training_info.task == 'TEXT_CLASSIFICATION':
                vectorize_path = str(modelpath).replace('tpot_predictor.pkl','text_vectorize.pkl')
                self.text_vectorizer = joblib.load(vectorize_path)

                feature_df = self.preprocess(df,training_info)
                y_pred = model.predict(feature_df)
                print('Y_pred: ',y_pred)
            return y_pred
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"An unexpected error occurred: {e}")
            return None
    
    def postprocess(self, data, training_info: TrainingInfo):
        pass

    def save_pipeline(self, pipeline_path: Path):
        try:
            if self.model is None:
                raise ValueError("Model is not trained")
            
            pipeline_path = Path(pipeline_path)
            pipeline_path.mkdir(parents=True, exist_ok=True)
            
            pipeline_file = pipeline_path / "pipeline.py"
            self.model.export(str(pipeline_file))
            
            self._logger.info(f"Pipeline đã được lưu tại {pipeline_file}")
        except Exception as e:
            self._logger.error(f"Lỗi khi lưu pipeline: {e}")
            raise

if __name__ == "__main__":
    trainer = TPOTTrainer()
    print('Start train--->')
    trainer.train(
        TrainingInfo(
            task="IMAGE_CLASSIFICATION",
            task_id="1",
            framework="TPOT",
            training_time=60*60,
            presets="medium_quality",
            text_column="org",
            label_column="label",
            image_column="image",
        )
    )
    # trainer.predict('/home/quoc/works/Learn/NCKH/model/tpot_predictor.pkl',
    #                 '/home/quoc/works/Learn/NCKH/loan-approval-tabular-classification/test.csv',
    #                 TrainingInfo(
    #                     task="TEXT_CLASSIFICATION",
    #                     task_id="1",
    #                     framework="TPOT",
    #                     training_time=60,
    #                     presets="medium_quality",
    #                     text_column="sentence",
    #                     label_column='',
    #                     image_column="image",
    #                 ))