import torch
import argparse
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from modules.dataset import build_vocab_from_files, decode_batch
from modules.transformer import TransformerConditionalGeneration
from modules.config import ModelConfig, TrainingConfig  # TrainingConfig used for paths


class TestDataset(Dataset):
    """Dataset for inference: reads source sentences and encodes them."""
    def __init__(self, src_vocab, file_path_src, max_len=None):
        self.src_vocab = src_vocab
        self.src_pad = src_vocab["<pad>"]
        self.src_bos = src_vocab["<bos>"]
        self.src_eos = src_vocab["<eos>"]
        self.src_unk = src_vocab["<unk>"]

        with open(file_path_src, encoding="utf-8") as f:
            self.src_texts = [line.strip() for line in f]

        # Encode all sentences (without padding yet)
        self.encoded = [self.encode(t) for t in self.src_texts]

        # Determine max length for padding (either fixed or from data)
        if max_len is None:
            self.max_len = max(len(seq) for seq in self.encoded)
        else:
            self.max_len = max_len

    def encode(self, text):
        tokens = text.split()
        ids = [self.src_vocab.get(t, self.src_unk) for t in tokens]
        return [self.src_bos] + ids + [self.src_eos]

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        seq = self.encoded[idx]
        padded = torch.full((self.max_len,), self.src_pad, dtype=torch.long)
        padded[:len(seq)] = torch.tensor(seq, dtype=torch.long)
        return padded, len(seq)


def collate_test(batch):
    src, src_len = zip(*batch)
    src = torch.stack(src)
    src_len = torch.tensor(src_len)
    return src, src_len


def main():
    parser = argparse.ArgumentParser(description="Translate using trained Transformer model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the saved model checkpoint (.pt file)")
    parser.add_argument("--test_src", type=str, required=True,
                        help="Path to the source language test file (e.g., test.de-en.de)")
    parser.add_argument("--output", type=str, default="translations.txt",
                        help="Output file for translations")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for inference")
    parser.add_argument("--max_len", type=int, default=100,
                        help="Maximum generation length")
    parser.add_argument("--data_folder", type=str, default="data",
                        help="Folder containing the original training files (for vocab)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = build_vocab_from_files([
        f"{args.data_folder}/train.de-en.de",
        f"{args.data_folder}/train.de-en.en",
        f"{args.data_folder}/val.de-en.de",
        f"{args.data_folder}/val.de-en.en",
    ])
    model_config = ModelConfig()
    model_config.VOCAB_SIZE = len(vocab)
    model_config.PAD_TOKEN_ID = vocab["<pad>"]
    model_config.BOS_TOKEN_ID = vocab["<bos>"]
    model_config.EOS_TOKEN_ID = vocab["<eos>"]
    model = TransformerConditionalGeneration(model_config)
    state_dict = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    test_dataset = TestDataset(vocab, args.test_src, max_len=None)  # will use max of this dataset
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_test
    )
    all_translations = []
    with torch.no_grad():
        for src_batch, _ in tqdm(test_loader, desc="Translating"):
            src_batch = src_batch.to(device)
            generated_ids = model.generate(src_batch, max_length=args.max_len)
            translations = decode_batch(
                generated_ids,
                vocab,
                pad_id=model_config.PAD_TOKEN_ID,
                eos_id=model_config.EOS_TOKEN_ID
            )
            all_translations.extend(translations)
    with open(args.output, "w", encoding="utf-8") as f:
        for line in all_translations:
            f.write(line + "\n")

    print(f"Translations saved to {args.output}")


if __name__ == "__main__":
    main()