from dataclasses import dataclass
import torch

@dataclass
class ModelConfig:
    NUM_HEADS: int = 16
    DIM_MODEL: int = 1024
    D_FF: int = 4096
    NUM_ENCODER_LAYERS: int = 4
    NUM_DECODER_LAYERS: int = 4
    DROPOUT: float = 0.5
    VOCAB_SIZE: int = 32000
    PAD_TOKEN_ID: int = 0
    BOS_TOKEN_ID: int = 2
    EOS_TOKEN_ID: int = 3
    MAX_SEQ_LEN: int = 512

@dataclass
class TrainConfig:
    BATCH_SIZE: int = 20
    LR: float = 3e-3
    NUM_EPOCHS: int = 40
    DEVICE: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DATA_FOLDER: str = "./data"
    LOG_WANDB: bool = False
    USE_BF16: bool = False
    COMPILE: bool = True
    SAVE_PATH: str = "models/best_model.pt"
    GRAD_ACCUM_STEPS: int = 4

@dataclass
class InferenceConfig:
    MODEL_PATH: str = "best_model.pt"
    TEST_FILE: str = "./data/test1.de-en.de"
    OUTPUT_FILE: str = "./data/translations.txt"
    BATCH_SIZE: int = 32
    MAX_LEN: int = 100