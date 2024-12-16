import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import List, Dict, Union
import torch

from utils import load_config

class DataLoader(Dataset):
    def __init__(self, data_config, df: pd.DataFrame, stride: int = 1, inference: bool = False) -> None:
        self.data_config = data_config
        self.inference = inference
        self.df = df
        
        self.column_names = self.df.filter(regex='^P\\d+$').columns.tolist() # P1, P2.. 꼴의 컬럼만 추출 (P1_flag 이런건 제외)
        self.file_ids = self.df['file_id'].values if 'file_id' in df.columns else None # test 데이터의 경우 (추론 모드에서만)
        
        if inference: # test일 경우, (단일 시퀀스 데이터로) 테스트 데이터 준비시키기
            self._prepare_inference_data()
        else: # train일 경우, 슬라이딩 윈도우 방식으로 학습 데이터 준비시키기
            self._prepare_training_data(stride) 
            
    def _normalize_columns(self, data: np.ndarray) -> np.ndarray: # data 인자에 모든 시간에서의 압력값 리스트가 들어가는건가?
        """벡터화된 열 정규화"""
        mins = data.min(axis=0, keepdims=True)
        maxs = data.max(axis=0, keepdims=True)
        
        # mins와 maxs가 같으면 전체를 0으로 반환
        is_constant = (maxs == mins)
        if np.any(is_constant):
            normalized_data = np.zeros_like(data)
            normalized_data[:, is_constant.squeeze()] = 0
            return normalized_data
        
        # 정규화 수행
        return (data - mins) / (maxs - mins)
    
    def _prepare_inference_data(self) -> None:
        self.values = self.df[self.column_names].values.astype(np.float32) # P1, P2 ..컬럼에 대한 값들을 np배열에 저장
        '''(결과 예시)
        array([
            [10., 100., 5.],  -> P1의 값들
            [20., 200., 10.], -> P2의 값들
            [30., 300., 15.], ...
            [40., 400., 20.]
        ], dtype=float32)
        '''
        self.normalized_values = self._normalize_columns(self.values)
        '''(결과 예시)
        _normalize_colums 메서드의 return 형태 : (data - mins) / (maxs - mins) 
        계산 과정:
        [[(10-10)/(40-10), (100-100)/(400-100), (5-5)/(20-5)],
        [(20-10)/(40-10), (200-100)/(400-100), (10-5)/(20-5)],
        [(30-10)/(40-10), (300-100)/(400-100), (15-5)/(20-5)],
        [(40-10)/(40-10), (400-100)/(400-100), (20-5)/(20-5)]] 
        (그냥 0~1사이의 값으로 정규화하는 과정)
        array([
            [0.0, 0.0, 0.0],
            [0.33333334, 0.33333334, 0.33333334],
            [0.6666667, 0.6666667, 0.6666667],
            [1.0, 1.0, 1.0]
        ], dtype=float32)
        '''

    def _prepare_training_data(self, stride: int) -> None:
        self.values = self.df[self.column_names].values.astype(np.float32)
        
        # 시작 인덱스 계산 (stride 적용) -> 겹치는 슬라이딩 윈도우 아니고, stride만큼 띄어서 겹치지 않으면서 윈도우 생성
        potential_starts = np.arange(0, len(self.df) - self.data_config["WINDOW_GIVEN"], stride)
        
        # 각 윈도우의 마지막 다음 지점(window_size + 1)이 사고가 없는(0) 경우만 필터링
        accident_labels = self.df['anomaly'].values
        valid_starts = [ # 유효한 시작점인지 체크
            idx for idx in potential_starts 
            if idx + self.data_config["WINDOW_GIVEN"] < len(self.df) and  # 1. 범위 체크
            accident_labels[idx + self.data_config["WINDOW_GIVEN"]] == 0  # 2. 윈도우 다음 지점 체크
        ]
        self.start_idx = np.array(valid_starts)
        
        # 유효한 윈도우들만 추출하여 정규화
        windows = np.array([
            self.values[i:i + self.data_config["WINDOW_GIVEN"]] 
            for i in self.start_idx
        ])
        
        # (윈도우 수, 윈도우 크기, 특성 수)로 한번에 정규화
        self.input_data = np.stack([
            self._normalize_columns(window) for window in windows
        ])
        
    def __len__(self) -> int:
        if self.inference:
            return len(self.column_names)
        return len(self.start_idx) * len(self.column_names)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[str, torch.Tensor]]:
        if self.inference:
            col_idx = idx
            col_name = self.column_names[col_idx]
            col_data = self.normalized_values[:, col_idx]
            file_id = self.file_ids[idx] if self.file_ids is not None else None
            return {
                "column_name": col_name,
                "input": torch.from_numpy(col_data).unsqueeze(-1),  # (time_steps, 1)
                "file_id": file_id
            }
        
        window_idx = idx // len(self.column_names)
        col_idx = idx % len(self.column_names)
        
        return {
            "column_name": self.column_names[col_idx],
            "input": torch.from_numpy(self.input_data[window_idx, :, col_idx]).unsqueeze(-1)
        }