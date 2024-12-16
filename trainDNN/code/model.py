import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from utils import load_config

class Model_Handler(nn.Module):
    def __init__(self, model_config):
        self.model_config = model_config
        super(Model_Handler, self).__init__()

        # LSTM feature extractor
        self.lstm_feature = nn.LSTM(
            input_size=1, 
            hidden_size=self.model_config["HIDDEN_DIM_LSTM"],
            num_layers=self.model_config["NUM_LAYERS"],
            batch_first=True,
            dropout=self.model_config["DROPOUT"] if self.model_config["NUM_LAYERS"] > 1 else 0
        )

        # Encoder modules
        self.encoder = nn.Sequential(
            nn.Linear(self.model_config["HIDDEN_DIM_LSTM"], self.model_config["HIDDEN_DIM_LSTM"]//4), # 1024 -> 256
            nn.ReLU(),
            nn.Linear(self.model_config["HIDDEN_DIM_LSTM"]//4, self.model_config["HIDDEN_DIM_LSTM"]//8), # 256 -> 128
            nn.ReLU(),
        )

        # Decoder modules
        self.decoder = nn.Sequential(
            nn.Linear(self.model_config["HIDDEN_DIM_LSTM"]//8, self.model_config["HIDDEN_DIM_LSTM"]//4), # 128 -> 256
            nn.ReLU(),
            nn.Linear(self.model_config["HIDDEN_DIM_LSTM"]//4, self.model_config["HIDDEN_DIM_LSTM"]), # 256 -> 1024
        )

    def forward(self, x):
        _, (hidden, _) = self.lstm_feature(x)
        last_hidden = hidden[-1]  # (batch, hidden_dim)
        
        # AE
        latent_z = self.encoder(last_hidden)
        reconstructed_hidden = self.decoder(latent_z)
        # print(f"last_hidden.requires_grad: {last_hidden.requires_grad}") # [True]
        # print(f"latent_z.requires_grad: {latent_z.requires_grad}")  # Encoder 출력 : [True]
        # print(f"reconstructed_hidden.requires_grad: {reconstructed_hidden.requires_grad}")  # LSTM 출력 : [True]
        
        # # Gradient Checkpointing
        # x.requires_grad_()
        # def lstm_forward(x):
        #     _, (hidden, _) = self.lstm_feature(x)
        #     print(f"LSTM hidden.requires_grad : {hidden[-1].requires_grad}")
        #     return hidden[-1]
        # def encoder_forward(x):
        #     print(f"Encoder input.requires_grad : {x.requires_grad}")
        #     return self.encoder(x)
        # def decoder_forward(x):
        #     print(f"Decoder input.requires_grad : {x.requires_grad}")
        #     return self.decoder(x)
        
        # last_hidden = checkpoint(lstm_forward, x)
        # print(f"last_hidden.requires_grad: {last_hidden.requires_grad}")  # 확인

        # latent_z = checkpoint(encoder_forward, last_hidden)
        # print(f"latent_z.requires_grad: {latent_z.requires_grad}")  # 확인

        # reconstructed_hidden = checkpoint(decoder_forward, latent_z)
        
        return last_hidden, reconstructed_hidden