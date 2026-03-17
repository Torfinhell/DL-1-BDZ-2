import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

class TransformerMT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.shared = nn.Embedding(config.VOCAB_SIZE, config.DIM_MODEL)
        self.pos_embedding = nn.Embedding(config.MAX_SEQ_LEN, config.DIM_MODEL)
        self.dropout = nn.Dropout(config.DROPOUT)
        self.transformer = nn.Transformer(
            d_model=config.DIM_MODEL,
            nhead=config.NUM_HEADS,
            num_encoder_layers=config.NUM_ENCODER_LAYERS,
            num_decoder_layers=config.NUM_DECODER_LAYERS,
            dim_feedforward=config.D_FF,
            dropout=config.DROPOUT,
            activation='relu',
            batch_first=True,
            norm_first=True
        )
        self.lm_head = nn.Linear(config.DIM_MODEL, config.VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.shared.weight

    def _shift_right(self, labels):
        shifted = labels.new_zeros(labels.shape)
        shifted[..., 1:] = labels[..., :-1].clone()
        shifted[..., 0] = self.config.BOS_TOKEN_ID
        return shifted

    def _generate_causal_mask(self, size, device):
        return torch.triu(torch.ones(size, size, device=device) * float('-inf'), diagonal=1)

    def forward(self, input_ids, labels=None):
        src_padding_mask = input_ids == self.config.PAD_TOKEN_ID
        src_pos = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        src_emb = self.shared(input_ids) + self.pos_embedding(src_pos)
        src_emb = self.dropout(src_emb)

        if labels is not None:
            tgt_input = self._shift_right(labels)
            tgt_padding_mask = tgt_input == self.config.PAD_TOKEN_ID
            tgt_pos = torch.arange(tgt_input.size(1), device=tgt_input.device).unsqueeze(0)
            tgt_emb = self.shared(tgt_input) + self.pos_embedding(tgt_pos)
            tgt_emb = self.dropout(tgt_emb)

            causal_mask = self._generate_causal_mask(tgt_input.size(1), tgt_input.device)

            output = self.transformer(
                src_emb, tgt_emb,
                tgt_mask=causal_mask,
                src_key_padding_mask=src_padding_mask,
                tgt_key_padding_mask=tgt_padding_mask,
                memory_key_padding_mask=src_padding_mask
            )
            logits = self.lm_head(output)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1),
                                   ignore_index=self.config.PAD_TOKEN_ID)
            return loss, logits
        else:
            memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_padding_mask)
            return memory

    @torch.no_grad()
    def generate(self, input_ids, max_len=100):
        self.eval()
        src_padding_mask = input_ids == self.config.PAD_TOKEN_ID
        src_pos = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        src_emb = self.shared(input_ids) + self.pos_embedding(src_pos)
        src_emb = self.dropout(src_emb)
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_padding_mask)

        batch_size = input_ids.size(0)
        dec_input = torch.full((batch_size, 1), self.config.BOS_TOKEN_ID, dtype=torch.long, device=input_ids.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

        for _ in range(max_len):
            tgt_pos = torch.arange(dec_input.size(1), device=dec_input.device).unsqueeze(0)
            tgt_emb = self.shared(dec_input) + self.pos_embedding(tgt_pos)
            tgt_emb = self.dropout(tgt_emb)
            causal_mask = self._generate_causal_mask(dec_input.size(1), dec_input.device)
            tgt_padding_mask = dec_input == self.config.PAD_TOKEN_ID

            output = self.transformer.decoder(
                tgt_emb, memory,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=tgt_padding_mask,
                memory_key_padding_mask=src_padding_mask
            )
            logits = self.lm_head(output[:, -1:, :])
            next_token = logits.argmax(dim=-1)
            dec_input = torch.cat([dec_input, next_token], dim=1)
            finished |= (next_token.squeeze(1) == self.config.EOS_TOKEN_ID)
            if finished.all():
                break
        return dec_input