import torch
from tqdm import tqdm

class Trainer():
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, trainer_config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.trainer_config = trainer_config
        
    def train_AE(self):
        train_losses = []
        val_losses = []
        best_model = {
            "loss": float('inf'),
            "state": None,
            "epoch": 0
        }
        
        # Train Loss 계산
        for epoch in range(self.trainer_config["EPOCHS"]):
            self.model.train()
            epoch_loss = 0.0
            '''
            tqdm : 진행률 바를 쉽게 추가할 수 있는 라이브러리
            e.g.
            for i in tqdm(range(100)):
                ...
            0부터 99까지 반복하면서, 진행률 바를 표시
            0%|          | 0/100 [00:00<?, ?it/s] 이런 식으로
            '''
            with tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.trainer_config['EPOCHS']}", unit="batch") as t:
                for batch in t: # 배치 단위로 train_data를 train_loader로 로드
                    # print(f"Batch['input']의 사이즈 : {batch['input'].shape}") # [32, 10080, 1] = [bs, window, feature수]
                    inputs = batch["input"].to(self.trainer_config["DEVICE"])
                    original_hidden, reconstructed_hidden = self.model(inputs) # [ Batch_size, HIDDEN_DIM_LSTM ]

                    loss = self.criterion(reconstructed_hidden, original_hidden)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step() # 파라미터 

                    epoch_loss += loss.item()
                    t.set_postfix(loss=loss.item())
                    # break

            avg_train_loss = epoch_loss / len(self.train_loader.dataset)
            train_losses.append(avg_train_loss)

            self.model.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                with tqdm(self.val_loader, desc=f"Epoch {epoch + 1}/{self.trainer_config['EPOCHS']} (Validation)", unit="batch") as t:
                    for batch in t:
                        inputs = batch["input"].to(self.trainer_config["DEVICE"])
                        original_hidden, reconstructed_hidden = self.model(inputs)

                        loss = self.criterion(reconstructed_hidden, original_hidden)
                        epoch_val_loss += loss.item()
                        t.set_postfix(loss=loss.item())
                        # break

            avg_val_loss = epoch_val_loss / len(self.val_loader.dataset)
            val_losses.append(avg_val_loss)

            print(f"Epoch {epoch + 1}/{self.trainer_config['EPOCHS']}, "
                  f"Average Train Loss: {avg_train_loss:.8f}, "
                  f"Average Validation Loss: {avg_val_loss:.8f}")

            # Save Best Model
            if avg_val_loss < best_model["loss"]:
                best_model["state"] = self.model.state_dict()
                best_model["loss"] = avg_val_loss
                best_model["epoch"] = epoch + 1

        return train_losses, val_losses, best_model