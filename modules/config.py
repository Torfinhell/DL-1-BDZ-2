from dataclasses import dataclass
import torch

@dataclass
class ModelConfig:
    NUM_HEADS:int=4
    DIM_KV:int=128
    DIM_MODEL:int=512
    EPS_LAYER_NORM:int=1e-8
    D_FF=2048
    NUM_ENCODER_LAYERS:int=4
    NUM_DECODER_LAYERS:int=4
    VOCAB_SIZE:int=0
    PAD_TOKEN_ID:int=0 
    BOS_TOKEN_ID:int=2
    EOS_TOKEN_ID:int=4
@dataclass
class TrainingConfig:
    BATCH_SIZE: int = 40
    DEVICE: torch.device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu'
    )
    DATA_FOLDER:str="data"
    NUM_EPOCHS:int=100
    LR:float=1e-4
    TRAIN_EPOCH_LEN:int=None
    GRAD_ACUM:int=4
@dataclass
class InferenceConfig:
    pass