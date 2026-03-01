import torch
from torch.utils.data import Dataset
from collections import Counter
from typing import Optional

class TranslationDataset(Dataset):
    def __init__(self, src_vocab, tgt_vocab, file_path_de, file_path_en, train_epoch_len:Optional[int]=None):
        """
        src_vocab: словарь для немецкого
        tgt_vocab: словарь для английского
        file_path_de: путь к .de файлу
        file_path_en: путь к .en файлу
        """
        self.train_epoch_len=train_epoch_len
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_pad = src_vocab["<pad>"]
        self.src_bos = src_vocab["<bos>"]
        self.src_eos = src_vocab["<eos>"]
        self.src_unk = src_vocab["<unk>"]

        self.tgt_pad = tgt_vocab["<pad>"]
        self.tgt_bos = tgt_vocab["<bos>"]
        self.tgt_eos = tgt_vocab["<eos>"]
        self.tgt_unk = tgt_vocab["<unk>"]
        with open(file_path_de, encoding="utf-8") as f:
            self.src_texts = [line.strip() for line in f]

        with open(file_path_en, encoding="utf-8") as f:
            self.tgt_texts = [line.strip() for line in f]

        assert len(self.src_texts) == len(self.tgt_texts), \
            "Source and target files must have the same number of lines"
        self.src_max_len = max(len(self.encode_src(t)) for t in self.src_texts)
        self.tgt_max_len = max(len(self.encode_tgt(t)) for t in self.tgt_texts)

    def __len__(self):
        if self.train_epoch_len is not None:
            return min(len(self.src_texts), self.train_epoch_len)
        return len(self.src_texts)

    def encode_src(self, text):
        tokens = text.split()
        ids = [self.src_vocab.get(t, self.src_unk) for t in tokens]
        return [self.src_bos] + ids + [self.src_eos]

    def encode_tgt(self, text):
        tokens = text.split()
        ids = [self.tgt_vocab.get(t, self.tgt_unk) for t in tokens]
        return [self.tgt_bos] + ids + [self.tgt_eos]

    def pad_sequence(self, seq, max_len, pad_index):
        padded = torch.full((max_len,), pad_index, dtype=torch.long)
        padded[:len(seq)] = torch.tensor(seq, dtype=torch.long)
        return padded

    def __getitem__(self, index):
        src_encoded = self.encode_src(self.src_texts[index])
        tgt_encoded = self.encode_tgt(self.tgt_texts[index])

        src_padded = self.pad_sequence(
            src_encoded, self.src_max_len, self.src_pad
        )

        tgt_padded = self.pad_sequence(
            tgt_encoded, self.tgt_max_len, self.tgt_pad
        )

        return (
            src_padded,
            tgt_padded,
            len(src_encoded),
            len(tgt_encoded),
        )

def build_vocab_from_files(file_paths, min_freq=1):
    """
    file_paths: список путей к файлам
    min_freq: минимальная частота токена
    return: vocab (dict token -> index)
    """
    
    counter = Counter()
    for path in file_paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split()
                counter.update(tokens)
    vocab = {
        "<pad>": 0,
        "<unk>": 1,
        "<bos>": 2,
        "<eos>": 3,
    }

    for token, freq in counter.items():
        if freq >= min_freq:
            vocab[token] = len(vocab)
    return vocab
def collate_fn(batch):
    src, tgt, src_len, tgt_len = zip(*batch)

    src = torch.stack(src)        
    tgt = torch.stack(tgt)        

    src_len = torch.tensor(src_len)
    tgt_len = torch.tensor(tgt_len)

    return src, tgt, src_len, tgt_len
def decode_batch(batch_ids, vocab, pad_id, eos_id):
    id2token = {v: k for k, v in vocab.items()}
    sentences = []

    for seq in batch_ids:
        tokens = []
        for idx in seq.tolist():

            if idx == eos_id:
                break

            if idx in (pad_id, vocab["<bos>"]):
                continue

            tokens.append(id2token.get(idx, "<unk>"))

        sentences.append(" ".join(tokens))

    return sentences
