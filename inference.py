import torch
import argparse
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from modules.dataset import build_vocab_from_files, TranslationDataset, collate_fn, decode_batch
from modules.transformer import TransformerConditionalGeneration
from modules.config import ModelConfig

def main():
    parser = argparse.ArgumentParser(description="Translate using a trained Transformer model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model checkpoint (.pt file)")
    parser.add_argument("--test_file", type=str, required=True,
                        help="Path to the source test file (e.g., data/test1.de-en.de)")
    parser.add_argument("--output", type=str, default="translations.txt",
                        help="Output file for translations")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for inference")
    parser.add_argument("--max_len", type=int, default=100,
                        help="Maximum generation length (number of tokens)")
    parser.add_argument("--data_folder", type=str, default="data",
                        help="Folder containing the original training/validation files (used to rebuild vocabulary)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = build_vocab_from_files([
        f"{args.data_folder}/train.de-en.de",
        f"{args.data_folder}/train.de-en.en",
        f"{args.data_folder}/val.de-en.de",
        f"{args.data_folder}/val.de-en.en",
    ])
    config = ModelConfig()
    config.VOCAB_SIZE = len(vocab)
    config.PAD_TOKEN_ID = vocab["<pad>"]
    config.BOS_TOKEN_ID = vocab["<bos>"]
    config.EOS_TOKEN_ID = vocab["<eos>"]
    model = TransformerConditionalGeneration(config)
    state_dict = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    test_dataset = TranslationDataset(
        src_vocab=vocab,
        tgt_vocab=vocab,
        file_path_de=args.test_file,
        file_path_en=None,
        train_epoch_len=None 
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    all_translations = []
    with torch.no_grad():
        for src, tgt, src_len, tgt_len in tqdm(test_loader, desc="Translating"):
            src = src.to(device)
            generated_ids = model.generate(src, max_length=args.max_len)
            translations = decode_batch(
                generated_ids,
                vocab,
                pad_id=config.PAD_TOKEN_ID,
                eos_id=config.EOS_TOKEN_ID
            )
            all_translations.extend(translations)

    with open(args.output, "w", encoding="utf-8") as f:
        for line in all_translations:
            f.write(line + "\n")

    print(f"Translations saved to {args.output}")

if __name__ == "__main__":
    main()