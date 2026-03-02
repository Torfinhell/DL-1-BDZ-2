from dataclasses import dataclass
import torch

@dataclass
class ModelConfig:
    NUM_HEADS: int = 12
    DIM_KV: int = 64
    DIM_MODEL: int = 768
    EPS_LAYER_NORM: float = 1e-6
    D_FF: int = 3072
    NUM_ENCODER_LAYERS: int = 12
    NUM_DECODER_LAYERS: int = 12
    MAX_SEQ_LEN: int = 512
    DROPOUT: float = 0.1
    VOCAB_SIZE: int = 0
    PAD_TOKEN_ID: int = 0
    BOS_TOKEN_ID: int = 2
    EOS_TOKEN_ID: int = 4
    RELATIVE_ATTENTION_NUM_BUCKETS: int = 32

@dataclass
class TrainingConfig:
    BATCH_SIZE: int = 16
    DEVICE: torch.device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu'
    )
    DATA_FOLDER: str = "data"
    NUM_EPOCHS: int = 6
    LR: float = 1e-4
    TRAIN_EPOCH_LEN: int = None
    GRAD_ACUM: int = 4

@dataclass
class InferenceConfig:
    pass