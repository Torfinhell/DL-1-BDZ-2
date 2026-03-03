import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import sentencepiece as spm
from functools import partial
from modules.transformer import TransformerConditionalGeneration
from modules.config import ModelConfig, InferenceConfig
from modules.dataset import TranslationDataset, collate_fn, decode_batch
from modules.post_processing import remove_duplicate_tokens, convert_to_list

def inference(inference_config: InferenceConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src_sp = spm.SentencePieceProcessor()
    src_sp.load("spm_de.model")
    tgt_sp = spm.SentencePieceProcessor()
    tgt_sp.load("spm_en.model")

    config = ModelConfig()
    config.VOCAB_SIZE = max(src_sp.get_piece_size(), tgt_sp.get_piece_size())
    config.PAD_TOKEN_ID = src_sp.pad_id()
    config.BOS_TOKEN_ID = src_sp.bos_id()
    config.EOS_TOKEN_ID = src_sp.eos_id()

    model = TransformerConditionalGeneration(config)
    state_dict = torch.load(inference_config.MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    test_dataset = TranslationDataset(
        src_sp=src_sp,
        tgt_sp=tgt_sp,
        src_file=inference_config.TEST_FILE,
        tgt_file=inference_config.TEST_FILE,   
        train_epoch_len=None
    )

    collate_fn_with_pad = partial(collate_fn, pad_id=config.PAD_TOKEN_ID)
    test_loader = DataLoader(
        test_dataset,
        batch_size=inference_config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn_with_pad
    )

    all_translations = []
    with torch.no_grad():
        for src, _ in tqdm(test_loader, desc="Translating"):
            src = src.to(device)
            generated_ids = model.generate(src, max_length=inference_config.MAX_LEN)
            cleaned_sequences = remove_duplicate_tokens(generated_ids)
            # cleaned_sequences=convert_to_list(generated_ids)
            for seq in cleaned_sequences:
                filtered = [t for t in seq if t not in (config.PAD_TOKEN_ID, config.EOS_TOKEN_ID, config.BOS_TOKEN_ID)]
                text = tgt_sp.decode(filtered)
                all_translations.append(text)


    with open(inference_config.OUTPUT_FILE, "w", encoding="utf-8") as f:
        for line in all_translations:
            f.write(line + "\n")
    print(f"Translations saved to {inference_config.OUTPUT_FILE}")

if __name__ == "__main__":
    inference_config = InferenceConfig()
    inference(inference_config)