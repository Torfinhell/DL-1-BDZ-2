import torch
from torch.utils.data import Dataset
from typing import Optional, List, Tuple
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tokenizers import PreTokenizer, NormalizedString, PreTokenizedString
import spacy

# ---------- Custom spaCy PreTokenizer ----------
class SpacyPreTokenizer(PreTokenizer):
    def __init__(self, lang: str):
        self.nlp = spacy.load(lang)

    def _split_using_spacy(self, i: int, normalized: NormalizedString) -> List[Tuple[str, (int, int)]]:
        text = str(normalized)
        doc = self.nlp(text)
        splits = []
        for token in doc:
            start = token.idx
            end = token.idx + len(token.text)
            splits.append((token.text, (start, end)))
        return splits

    def pre_tokenize(self, pretok: PreTokenizedString):
        pretok.split(self._split_using_spacy)
# ------------------------------------------------

def build_tokenizer(
    file_paths: List[str],
    spacy_lang: str,
    vocab_size: int = 32000,
    min_freq: int = 1
) -> Tokenizer:
    """Train a BPE tokenizer on given files with spaCy pre‑tokenization."""
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = SpacyPreTokenizer(spacy_lang)
    tokenizer.decoder = decoders.BPE()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_freq,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"]
    )
    tokenizer.train(file_paths, trainer)
    return tokenizer


class TranslationDataset(Dataset):
    def __init__(
        self,
        src_tokenizer: Tokenizer,
        tgt_tokenizer: Tokenizer,
        src_file: str,
        tgt_file: str,
        train_epoch_len: Optional[int] = None
    ):
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.train_epoch_len = train_epoch_len

        with open(src_file, encoding="utf-8") as f:
            self.src_texts = [line.strip() for line in f]
        with open(tgt_file, encoding="utf-8") as f:
            self.tgt_texts = [line.strip() for line in f]
        assert len(self.src_texts) == len(self.tgt_texts)
        self.src_pad = src_tokenizer.token_to_id("<pad>")
        self.src_bos = src_tokenizer.token_to_id("<bos>")
        self.src_eos = src_tokenizer.token_to_id("<eos>")
        self.src_unk = src_tokenizer.token_to_id("<unk>")

        self.tgt_pad = tgt_tokenizer.token_to_id("<pad>")
        self.tgt_bos = tgt_tokenizer.token_to_id("<bos>")
        self.tgt_eos = tgt_tokenizer.token_to_id("<eos>")
        self.tgt_unk = tgt_tokenizer.token_to_id("<unk>")
        self.src_max_len = max(len(self._encode_src(t)) for t in self.src_texts)
        self.tgt_max_len = max(len(self._encode_tgt(t)) for t in self.tgt_texts)

    def __len__(self):
        if self.train_epoch_len is not None:
            return min(len(self.src_texts), self.train_epoch_len)
        return len(self.src_texts)

    def _encode_src(self, text: str) -> List[int]:
        encoding = self.src_tokenizer.encode(text)
        return [self.src_bos] + encoding.ids + [self.src_eos]

    def _encode_tgt(self, text: str) -> List[int]:
        encoding = self.tgt_tokenizer.encode(text)
        return [self.tgt_bos] + encoding.ids + [self.tgt_eos]

    def pad_sequence(self, seq: List[int], max_len: int, pad_idx: int) -> torch.Tensor:
        padded = torch.full((max_len,), pad_idx, dtype=torch.long)
        padded[:len(seq)] = torch.tensor(seq, dtype=torch.long)
        return padded

    def __getitem__(self, idx):
        src_enc = self._encode_src(self.src_texts[idx])
        tgt_enc = self._encode_tgt(self.tgt_texts[idx])
        src_pad = self.pad_sequence(src_enc, self.src_max_len, self.src_pad)
        tgt_pad = self.pad_sequence(tgt_enc, self.tgt_max_len, self.tgt_pad)
        return src_pad, tgt_pad, len(src_enc), len(tgt_enc)


def collate_fn(batch):
    src, tgt, src_len, tgt_len = zip(*batch)
    return torch.stack(src), torch.stack(tgt), torch.tensor(src_len), torch.tensor(tgt_len)


def decode_batch(batch_ids, tokenizer: Tokenizer, pad_id: int, eos_id: int) -> List[str]:
    sentences = []
    bos_id = tokenizer.token_to_id("<bos>")
    for seq in batch_ids:
        ids = [idx for idx in seq.tolist() if idx not in (pad_id, eos_id, bos_id)]
        sentences.append(tokenizer.decode(ids, skip_special_tokens=True))
    return sentences    