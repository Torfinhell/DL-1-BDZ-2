import torch
from torch.utils.data import Dataset
from typing import Optional, List
import sentencepiece as spm
import os

def train_sentencepiece(
    file_paths: List[str],
    model_prefix: str,
    vocab_size: int = 32000,
    model_type: str = "bpe"
) -> spm.SentencePieceProcessor:
    spm.SentencePieceTrainer.train(
        input=file_paths,               
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=1.0,
        add_dummy_prefix=False,
        remove_extra_whitespaces=False,
        hard_vocab_limit=False,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_id=0,
        unk_piece="<unk>",
        bos_piece="<bos>",
        eos_piece="<eos>",
        pad_piece="<pad>"
    )
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")
    return sp


class TranslationDataset(Dataset):
    def __init__(
        self,
        src_sp: spm.SentencePieceProcessor,
        tgt_sp: spm.SentencePieceProcessor,
        src_file: str,
        tgt_file: str,
        train_epoch_len: Optional[int] = None,
        max_seq_len: Optional[int] = None          # new parameter
    ):
        """
        Args:
            src_sp: Source language SentencePiece processor.
            tgt_sp: Target language SentencePiece processor.
            src_file: Path to source text file (one sentence per line).
            tgt_file: Path to target text file (aligned with src_file).
            train_epoch_len: If set, limits dataset size (useful for debugging).
            max_seq_len: Maximum allowed sequence length (including BOS/EOS). 
                         Pairs exceeding this length are filtered out.
        """
        self.src_sp = src_sp
        self.tgt_sp = tgt_sp
        self.train_epoch_len = train_epoch_len
        self.max_seq_len = max_seq_len

        # Read all lines
        with open(src_file, encoding="utf-8") as f:
            src_texts = [line.strip() for line in f]
        with open(tgt_file, encoding="utf-8") as f:
            tgt_texts = [line.strip() for line in f]
        assert len(src_texts) == len(tgt_texts)

        # Store tokenizer special IDs
        self.src_pad = src_sp.pad_id()
        self.src_bos = src_sp.bos_id()
        self.src_eos = src_sp.eos_id()
        self.src_unk = src_sp.unk_id()

        self.tgt_pad = tgt_sp.pad_id()
        self.tgt_bos = tgt_sp.bos_id()
        self.tgt_eos = tgt_sp.eos_id()
        self.tgt_unk = tgt_sp.unk_id()

        # Filter out pairs that exceed max_seq_len (if specified)
        self.src_texts = []
        self.tgt_texts = []
        max_src_len = 0
        max_tgt_len = 0

        for src_text, tgt_text in zip(src_texts, tgt_texts):
            src_enc = self._encode_src(src_text)
            tgt_enc = self._encode_tgt(tgt_text)

            if max_seq_len is not None:
                if len(src_enc) > max_seq_len or len(tgt_enc) > max_seq_len:
                    continue  # skip this pair

            self.src_texts.append(src_text)
            self.tgt_texts.append(tgt_text)

            # Update max lengths for padding
            if len(src_enc) > max_src_len:
                max_src_len = len(src_enc)
            if len(tgt_enc) > max_tgt_len:
                max_tgt_len = len(tgt_enc)

        if len(self.src_texts) == 0:
            raise RuntimeError("No sentences left after filtering with max_seq_len={}".format(max_seq_len))

        # Set padding lengths to the actual maximums (which are ≤ max_seq_len if filtering was applied)
        self.src_max_len = max_src_len
        self.tgt_max_len = max_tgt_len

    def __len__(self):
        if self.train_epoch_len is not None:
            return min(len(self.src_texts), self.train_epoch_len)
        return len(self.src_texts)

    def _encode_src(self, text: str) -> List[int]:
        ids = self.src_sp.encode(text, out_type=int)
        return [self.src_bos] + ids + [self.src_eos]

    def _encode_tgt(self, text: str) -> List[int]:
        ids = self.tgt_sp.encode(text, out_type=int)
        return [self.tgt_bos] + ids + [self.tgt_eos]

    def pad_sequence(self, seq: List[int], max_len: int, pad_idx: int) -> torch.Tensor:
        padded = torch.full((max_len,), pad_idx, dtype=torch.long)
        padded[:len(seq)] = torch.tensor(seq, dtype=torch.long)
        return padded

    def __getitem__(self, idx):
        src_enc = self._encode_src(self.src_texts[idx])
        tgt_enc = self._encode_tgt(self.tgt_texts[idx])
        src_pad = self.pad_sequence(src_enc, self.src_max_len, self.src_pad)
        tgt_pad = self.pad_sequence(tgt_enc, self.tgt_max_len, self.tgt_pad)
        return src_pad, tgt_pad


def collate_fn(batch):
    src, tgt = zip(*batch)
    return torch.stack(src), torch.stack(tgt)


def decode_batch(batch_ids, sp_model: spm.SentencePieceProcessor, pad_id: int, eos_id: int) -> List[str]:
    sentences = []
    bos_id = sp_model.bos_id()
    for seq in batch_ids:
        ids = [idx for idx in seq.tolist() if idx not in (pad_id, eos_id, bos_id)]
        text = sp_model.decode(ids)
        sentences.append(text)
    return sentences