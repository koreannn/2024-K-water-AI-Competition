import os
from datetime import datetime

import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score
import ast

from data_loader import DataLoader
from utils import load_config

class Inference():
    def __init__(self, model, threshold, inference_config):
        self.model = model
        self.threshold = threshold
        self.inference_config = inference_config
        
        
    def inference_test_files(self, batch):
        self.model.eval()
        with torch.no_grad():
            inputs = batch["input"].to(self.inference_config["DEVICE"])
            original_hidden, reconstructed_hidden = self.model(inputs)
            reconstruction_loss = torch.mean((original_hidden - reconstructed_hidden) ** 2, dim=1).cpu().numpy()
        return reconstruction_loss


    def detect_anomaly(self, test_directory):
        test_files = [f for f in os.listdir(test_directory) if f.startswith("TEST") and f.endswith(".csv")]
        test_datasets = []
        all_test_data = []

        for filename in tqdm(test_files, desc='Processing test files'):
            test_file = os.path.join(test_directory, filename)
            df = pd.read_csv(test_file)
            df['file_id'] = filename.replace('.csv', '')
            individual_df = df[['timestamp', 'file_id'] + df.filter(like='P').columns.tolist()]
            individual_dataset = DataLoader(self.inference_config, individual_df, inference=True)
            test_datasets.append(individual_dataset)
            
            all_test_data.append(df)

        combined_dataset = torch.utils.data.ConcatDataset(test_datasets)

        test_loader = torch.utils.data.DataLoader(
            combined_dataset,
            batch_size=256,
            shuffle=False
        )

        reconstruction_errors = []
        for batch in tqdm(test_loader):
            reconstruction_loss = self.inference_test_files(batch)
            
            for i in range(len(reconstruction_loss)):
                reconstruction_errors.append({
                    "ID": batch["file_id"][i],
                    "column_name": batch["column_name"][i],
                    "reconstruction_error": reconstruction_loss[i]
                })
        
        errors_df = pd.DataFrame(reconstruction_errors)
        
        flag_columns = []
        for column in sorted(errors_df['column_name'].unique()):
            flag_column = f'{column}_flag'
            errors_df[flag_column] = (errors_df.loc[errors_df['column_name'] == column, 'reconstruction_error'] > self.threshold).astype(int)
            flag_columns.append(flag_column)

        errors_df_pivot = errors_df.pivot_table(index='ID', 
                                            columns='column_name', 
                                            values=flag_columns, 
                                            aggfunc='first')
        errors_df_pivot.columns = [f'{col[1]}' for col in errors_df_pivot.columns]
        errors_df_flat = errors_df_pivot.reset_index()

        errors_df_flat['flag_list'] = errors_df_flat.loc[:, 'P1':'P' + str(len(flag_columns))].apply(lambda x: x.tolist(), axis=1).apply(lambda x: [int(i) for i in x])
        return errors_df_flat[["ID", "flag_list"]]
        
        
    # def calculate_pressure_level_f1(self, gt_df, pred_file_name): # 경로를 전달 -> 내부에서 데이터프레임으로 변환 하는 과정으로 변경
    def get_f1_score(self, gt_df, pred_df): 
        
        pred_flags = pred_df['flag_list'].apply(lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else np.array(x))
        gt_flags = gt_df['flag_list'].apply(lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else np.array(x))
        '''
        ast.literal_eval(x) : ast(Abstract Syntax Tree)모듈에 있는 함수로, 
        문자열로 표현된 리터럴 데이터를 파이썬 객체로 변환하주는 역할
        
        e.g. 
        data = "[1, 2, 3, 4]" # type : str
        result = ast.literal_eval(data_str)
        data = [1, 2, 3, 4] 가 됨 (type : list)
        '''
        # 압력계 별 배점 가중치는 평가용으로 비공개
        # 단, 실제 답(GT)과 크게 다르지 않으며 정답에 근접한 예측에 대해 가산점을 주기 위한 용도.
        # weight_flags = gt_df['weight_list'].apply(lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else np.array(x))

        total_f1 = 0
        valid_samples = 0

        for idx in range(len(gt_df)):
            gt_row = np.array(gt_flags.iloc[idx])
            pred_row = np.array(pred_flags.iloc[idx])
            # weights_row = np.array(weight_flags.iloc[idx])
            if len(gt_row) != len(pred_row):
                raise ValueError("예측한 압력계 개수가 샘플 관망 구조와 다릅니다.")

            is_normal_sample = np.all(gt_row == 0)
            # 1) 정상 샘플에 대한 계산
            # -> 정상 샘플을 정상으로 잘 예측한 경우에는 점수 계산에 포함하지 않음
            # -> 정상 샘플을 비정상으로 잘못 예측한 경우에는 0점으로 반영 (패널티)
            if is_normal_sample:
                if np.sum(pred_row) > 0:  # False Positives
                    valid_samples += 1  # Include in valid samples
                    total_f1 += 0  # Penalize False Positives
                continue  # Skip further calculations for normal samples

            # 2) 비정상 샘플에 대한 계산
            # 정답과 예측이 동시에 1인 위치의 가중치 합
            # matched_abnormal_weights = np.sum(weights_row * (gt_row == 1) * (pred_row == 1))
            matched_abnormal_weights = np.sum((gt_row == 1) * (pred_row == 1))
            # 예측값이 1인 위치의 가중치 합
            # predicted_abnormal_weights = np.sum(weights_row * (pred_row == 1))
            predicted_abnormal_weights = np.sum(pred_row == 1)
            # 정답 위치의 가중치 합
            # total_abnormal_weights = np.sum(weights_row * (gt_row == 1))
            total_abnormal_weights = np.sum(gt_row == 1)

            # False Positives: 정답이 0이고, 가중치도 0인데 예측이 1인 경우
            # false_positives = np.sum((pred_row == 1) & (gt_row == 0) & (weights_row == 0))
            false_positives = np.sum((pred_row == 1) & (gt_row == 0))

            # Precision 계산: False Positives를 고려한 방식
            precision = (matched_abnormal_weights / (predicted_abnormal_weights + false_positives) 
                        if (predicted_abnormal_weights + false_positives) > 0 else 0) # else None -> else 0
            recall = (matched_abnormal_weights / total_abnormal_weights
                    if total_abnormal_weights > 0 else 0) # else None -> else 0

            if precision is not None and recall is not None and precision + recall > 0:
                f1_score = 2 * (precision * recall) / (precision + recall)
            else:
                f1_score = 0

            total_f1 += f1_score
            valid_samples += 1

        average_f1 = total_f1 / valid_samples if valid_samples > 0 else 0
        return precision, recall, average_f1