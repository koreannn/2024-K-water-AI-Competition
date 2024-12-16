import os
from datetime import datetime
import warnings

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import Subset

from data_loader import DataLoader
from model import Model_Handler
from trainer import Trainer
from set_threshold import SetThreshold
from inference import Inference
from utils import load_config

def main():

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    # torch.set_printoptions(threshold=torch.inf)
    warnings.filterwarnings("ignore")
    config = load_config()
    
    df_A = pd.read_csv(config["train_data"]["train_A_path"]) # len(df_A) : 44101 (2024.05.27 00:00 ~ 2024.06.26 15:00)
    df_B = pd.read_csv(config["train_data"]["train_B_path"]) # len(df_B) : 41760 (2024.07.01 00:00 ~ 2024.07.29 23:59)

    # Create Dataset
    train_dataset_A = DataLoader(config, df_A, stride=60, inference=False)
    # print("df_A:\n", df_A.head())
    for batch in train_dataset_A: # DataLoader.py __getitem__()
        print(f"Len of Batch : {len(batch)}")
        print(f"Len of Batch['input'] : {len(batch['input'])}")
        print(f"Total train_dataset_A length : {len(train_dataset_A)}")
        break
            
    train_dataset_B = DataLoader(config, df_B, stride=60, inference=False)
    # print("df_B:\n", df_B.head())
    for batch in train_dataset_B: # DataLoader.py __getitem__()
        print(f"Len of Batch : {len(batch)}")
        print(f"Len of Batch['input'] : {len(batch['input'])}")
        print(f"Total train_dataset_B length : {len(train_dataset_B)}")
        break
    
    train_dataset_A_B = torch.utils.data.ConcatDataset([train_dataset_A, train_dataset_B])
    print(f"\n[len(train_dataset_A_B)] Total Dataset Length: {len(train_dataset_A_B)}\n")
    
    tscv = TimeSeriesSplit(n_splits=5) # val set split & 반환값은 리스트
    for train_idx, val_idx in tscv.split(range(len(train_dataset_A_B))):
        train_dataset = Subset(train_dataset_A_B, train_idx)
        val_dataset = Subset(train_dataset_A_B, val_idx)
    
    print(f"[Train] Train Dataset Length: {len(train_dataset)}")
    print(f"[Validation] Validation Dataset Length: {len(val_dataset)}")
    
    # torch.utils.data.DataLoader : 배치 처리, 셔플링, 병렬 처리, 데이터 변환(전처리 작업 시) 등을 지원
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                                batch_size=config["BATCH_SIZE"], 
                                                shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=config["BATCH_SIZE"],
                                                shuffle=False)
    # (확인용) train_loader, val_loader가 배치사이즈를 기반으로, 제대로 된 DataLoader의 반복횟수가 나오는지 확인
    print(f"[Train] Train Loader Length: {len(train_loader)}")
    print(f"[Validation] Validation Loader Length: {len(val_loader)}\n")
    
    for i, batch in enumerate(train_loader):
        print(f"[Train] Batch {i+1} / {len(train_loader)}")
        print(f"[Train] Batch['input'].shape : {batch['input'].shape}\n")
        break
    for i, batch in enumerate(val_loader):
        print(f"[Validation] Batch {i+1} / {len(val_loader)}")
        print(f"[Valdation] Batch['input'].shape : {batch['input'].shape}\n") # torch.Size([32, 10080, 1]) -> 배치당 데이터 샘플 수, 시퀀스 길이, 특성 차원
        break
        '''
        특성 차원(Feature Dimension) : 값이 1이라면 단변량 시계열 데이터
        예를 들어 센서 데이터를 다룬다고 할 경우, 온도 데이터만 기록된 경우
        
        온도, 습도, 풍속이라는 세 가지 변수를 기록한다면, 특성 차원 값은 3이 되어야 한다.
        '''
                                             
    
    model = Model_Handler(config).cuda()
    print("\n<Model Summary>")
    print(model)
    print(f"Total Model Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # 메모리 추적
    print(f"[Memory] Memory Tracing : {torch.cuda.memory_allocated()/(1024**2):.2f}MB")
    
    # [테스트] 테스트용 입력 데이터 샘플 하나 집어넣어보기
    sample_batch = next(iter(train_loader))
    inputs = sample_batch["input"].to(config["DEVICE"])
    # print(f"Batch Inputs : f{inputs}")
    last_hidden, reconstructed_hidden = model(inputs)
    # print(f"\nInput Shape: {inputs.shape}") # torch.Size([64, 10080, 1]) -> [Batch_size, Sequence_length, Feature_dim]
    # print(f"Last Hidden Shape: {last_hidden.shape}") # torch.Size([64, 128]) -> [Batch_size, Hidden_dim]
    # print(f"Reconstructed Hidden Shape: {reconstructed_hidden.shape}") # torch.Size([64, 128]) -> [Batch_size, Hidden_dim]

    criterion = nn.MSELoss()
    print("[Criterion]", criterion)
    optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])
    print("[Optimizer]", optimizer)

    print("\n\n<Training Start!>\n\n")
    train_losses, val_losses, best_model = Trainer(model=model, 
                                        train_loader=train_loader, 
                                        val_loader=val_loader,
                                        optimizer=optimizer, 
                                        criterion=criterion,
                                        trainer_config=config
                                        ).train_AE()
    print("Final Training Losses:", train_losses)
    print("Final Validation Losses:", val_losses)
    print("Best Model Loss:", best_model["loss"])
    print("Best Model Saved at Epoch:", best_model["epoch"])
    
    print(f"[Memory] Memory Tracing : {torch.cuda.memory_allocated()/(1024**2):.2f}MB")

    inference_model = model
    inference_model.load_state_dict(best_model["state"])
    # load_state_dict : 모델의 파라미터를 저장된 상태 딕셔너리로부터 로드해줌 - best_model["state"]에 저장된 파라미터값들이 model에 로드됨 (nn.Module로부터 상속받음)
    # best_model 제대로 저장되었는지 확인
    torch.save(best_model["state"], "best_model.pth")
    loaded_state = torch.load("best_model.pth")
    
    for name, param in loaded_state.items():
        if not torch.equal(param, best_model["state"][name]):
            raise ValueError("Model Loading Failed")
        else:
            print(f"Parameter {name} Loaded Successfully")

    print("\n\nInference Start!\n\n")
    THRESHOLD = SetThreshold(config, inference_model, train_loader).calculate_and_save_threshold()
    

    C_list = Inference(inference_model, THRESHOLD, config).detect_anomaly(test_directory=config["test_data"]["test_C_path"])
    print(f"C_list Length: {len(C_list)}") # C에 있는 .csv파일 개수
    print(f"C_list Sample\n: {C_list.head()}")
    D_list = Inference(inference_model, THRESHOLD, config).detect_anomaly(test_directory=config["test_data"]["test_D_path"])
    print(f"D_list Length: {len(D_list)}") # D에 있는 .csv파일 개수
    print(f"D_list Sample\n: {D_list.head()}")
    C_D_list = pd.concat([C_list, D_list])
    print(f"C_D_list Length: {len(C_D_list)}")
    print(f"C_D_list Sample\n: {C_D_list}")

    pred_submission = pd.read_csv(config["submission_template"]["submission_template_path"])
    # 매핑된 값으로 업데이트하되, 매핑되지 않은 경우 기존 값 유지
    flag_mapping = C_D_list.set_index("ID")["flag_list"]
    pred_submission["flag_list"] = pred_submission["ID"].map(flag_mapping).fillna(pred_submission["flag_list"])
    print(f"Sample Submission Length: {len(pred_submission)}")
    print(f"Sample Submission Sample (After Mapping)\n: {pred_submission.head()}")

    # output file 이름 지정 & csv 파일로 저장
    curr_time = datetime.now().strftime("%Y%m%d_%H%M")
    batch_size = config["BATCH_SIZE"]
    epochs = config["EPOCHS"]
    learning_rate = config["LEARNING_RATE"]
        
    file_name = f"{curr_time}_bs{batch_size}_ep{epochs}_lr{learning_rate}.csv"
    
    output_path_and_name = os.path.join(config["output_path"]["output_dir"], file_name)
    pred_submission.to_csv(output_path_and_name, index=False)
    print(f"Submission FIle Saved at {output_path_and_name}")
    print(f"Sample Rows from Saved Submission File:\n{pred_submission.head()}")
    
    precision, recall, f1_score = Inference(inference_model, THRESHOLD, config).get_f1_score(C_D_list, pred_submission)
    print(f"Precision : ", precision)
    print(f"Recall : ", recall)
    print(f"Average F1 Score : ", f1_score)
    
if __name__ == "__main__":
    main()