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
    ):
        self.src_sp = src_sp
        self.tgt_sp = tgt_sp
        self.train_epoch_len = train_epoch_len

        with open(src_file, encoding="utf-8") as f:
            src_texts = [line.strip() for line in f]
        with open(tgt_file, encoding="utf-8") as f:
            tgt_texts = [line.strip() for line in f]
        assert len(src_texts) == len(tgt_texts)
        self.src_pad = src_sp.pad_id()
        self.src_bos = src_sp.bos_id()
        self.src_eos = src_sp.eos_id()
        self.tgt_pad = tgt_sp.pad_id()
        self.tgt_bos = tgt_sp.bos_id()
        self.tgt_eos = tgt_sp.eos_id()

        self.src_texts = src_texts
        self.tgt_texts = tgt_texts

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

    def __getitem__(self, idx):
        src_enc = self._encode_src(self.src_texts[idx])
        tgt_enc = self._encode_tgt(self.tgt_texts[idx])
        return src_enc, tgt_enc, len(src_enc), len(tgt_enc)


def collate_fn(batch, pad_id: int):
    src_enc, tgt_enc, src_lens, tgt_lens = zip(*batch)
    max_src_len = max(src_lens)
    max_tgt_len = max(tgt_lens)

    src_padded = []
    tgt_padded = []
    for src, tgt in zip(src_enc, tgt_enc):
        src_pad = torch.full((max_src_len,), pad_id, dtype=torch.long)
        src_pad[:len(src)] = torch.tensor(src, dtype=torch.long)
        src_padded.append(src_pad)
        tgt_pad = torch.full((max_tgt_len,), pad_id, dtype=torch.long)
        tgt_pad[:len(tgt)] = torch.tensor(tgt, dtype=torch.long)
        tgt_padded.append(tgt_pad)
    return torch.stack(src_padded), torch.stack(tgt_padded)


def decode_batch(batch_ids, sp_model: spm.SentencePieceProcessor, pad_id: int, eos_id: int) -> List[str]:
    sentences = []
    bos_id = sp_model.bos_id()
    for seq in batch_ids:
        ids = [idx for idx in seq.tolist() if idx not in (pad_id, eos_id, bos_id)]
        text = sp_model.decode(ids)
        sentences.append(text)
    return sentences