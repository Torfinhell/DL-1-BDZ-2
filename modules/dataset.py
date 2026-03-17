import torch
from torch.utils.data import Dataset
from typing import Optional, List
import sentencepiece as spm
import os


def train_spm(files, model_prefix, vocab_size=32000):
    print("Starte")
    spm.SentencePieceTrainer.train(
        input=','.join(files), model_prefix=model_prefix, vocab_size=vocab_size,
        model_type='bpe', character_coverage=1.0,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        pad_piece='<pad>', unk_piece='<unk>', bos_piece='<bos>', eos_piece='<eos>',
    )
    sp = spm.SentencePieceProcessor()
    sp.load(f'{model_prefix}.model')
    return sp
class TranslationDataset(Dataset):
    def __init__(self, src_sp, tgt_sp, src_file, tgt_file):
        with open(src_file, encoding='utf-8') as f:
            self.src = [l.strip() for l in f]
        with open(tgt_file, encoding='utf-8') as f:
            self.tgt = [l.strip() for l in f]
        self.src_sp, self.tgt_sp = src_sp, tgt_sp

    def __len__(self):
        return len(self.src)

    def __getitem__(self, i):
        src_ids = [self.src_sp.bos_id()] + self.src_sp.encode(self.src[i]) + [self.src_sp.eos_id()]
        tgt_ids = [self.tgt_sp.bos_id()] + self.tgt_sp.encode(self.tgt[i]) + [self.tgt_sp.eos_id()]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)

def collate(batch, pad_id):
    src, tgt = zip(*batch)
    src = torch.nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=pad_id)
    tgt = torch.nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=pad_id)
    return src, tgt