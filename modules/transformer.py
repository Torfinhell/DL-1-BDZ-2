from torch import nn
import torch
from einops import rearrange
from typing import Optional

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class Attention(nn.Module):
    def __init__(self, d_model, num_heads, d_kv):
        super().__init__()
        self.num_heads = num_heads
        self.d_kv = d_kv
        self.inner_dim = num_heads * d_kv

        self.q = nn.Linear(d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, d_model, bias=False)

    def forward(
        self,
        hidden_states,
        key_value_states=None,
        mask: Optional[torch.Tensor] = None,
    ):
        source = key_value_states if key_value_states is not None else hidden_states

        q = rearrange(self.q(hidden_states), "B L (H D) -> B H L D", H=self.num_heads)
        k = rearrange(self.k(source), "B L (H D) -> B H L D", H=self.num_heads)
        v = rearrange(self.v(source), "B L (H D) -> B H L D", H=self.num_heads)

        scores = torch.matmul(q, k.transpose(-1, -2))
        scores = scores / (self.d_kv ** 0.5)

        if mask is not None:
            scores = scores + mask

        attn = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn, v)

        attn_output = rearrange(attn_output, "B H L D -> B L (H D)")
        return self.o(attn_output)


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = Attention(
            config.DIM_MODEL,
            config.NUM_HEADS,
            config.DIM_KV
        )
        self.layer_norm = RMSNorm(config.DIM_MODEL, eps=config.EPS_LAYER_NORM)
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, hidden_states, mask=None):
        normed = self.layer_norm(hidden_states)
        attn_out = self.attn(normed, mask=mask)
        return hidden_states + self.dropout(attn_out)


class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = Attention(
            config.DIM_MODEL,
            config.NUM_HEADS,
            config.DIM_KV
        )
        self.layer_norm = RMSNorm(config.DIM_MODEL, eps=config.EPS_LAYER_NORM)
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, hidden_states, encoder_hidden_states, mask=None):
        normed = self.layer_norm(hidden_states)
        attn_out = self.attn(
            normed,
            key_value_states=encoder_hidden_states,
            mask=mask
        )
        return hidden_states + self.dropout(attn_out)


class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm = RMSNorm(config.DIM_MODEL, eps=config.EPS_LAYER_NORM)
        self.wi = nn.Linear(config.DIM_MODEL, config.D_FF, bias=False)
        self.wo = nn.Linear(config.D_FF, config.DIM_MODEL, bias=False)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, hidden_states):
        normed = self.layer_norm(hidden_states)
        x = self.wi(normed)
        x = self.activation(x)
        x = self.wo(x)
        return hidden_states + self.dropout(x)


class TransformerBlock(nn.Module):
    def __init__(self, config, is_decoder):
        super().__init__()
        self.is_decoder = is_decoder

        self.self_attn = SelfAttention(config)

        if is_decoder:
            self.cross_attn = CrossAttention(config)

        self.ffn = FFN(config)

    def forward(
        self,
        hidden_states,
        self_mask=None,
        cross_mask=None,
        encoder_hidden_states=None
    ):
        hidden_states = self.self_attn(hidden_states, mask=self_mask)

        if self.is_decoder and encoder_hidden_states is not None:
            hidden_states = self.cross_attn(
                hidden_states,
                encoder_hidden_states,
                mask=cross_mask
            )

        hidden_states = self.ffn(hidden_states)
        return hidden_states


class TransformerEncoder(nn.Module):
    def __init__(self, config, shared_embedding):
        super().__init__()
        self.embed_tokens = shared_embedding
        self.pos_embed = nn.Embedding(config.MAX_SEQ_LEN, config.DIM_MODEL)
        self.embed_dropout = nn.Dropout(config.DROPOUT)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config, is_decoder=False)
             for _ in range(config.NUM_ENCODER_LAYERS)]
        )
        self.layer_norm = RMSNorm(config.DIM_MODEL, eps=config.EPS_LAYER_NORM)

    def forward(self, input_ids, padding_mask=None):
        seq_len = input_ids.size(1)
        token_emb = self.embed_tokens(input_ids)
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)
        pos_emb = self.pos_embed(positions)
        hidden_states = token_emb + pos_emb
        hidden_states = self.embed_dropout(hidden_states)

        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                self_mask=padding_mask
            )

        return self.layer_norm(hidden_states)


class TransformerDecoder(nn.Module):
    def __init__(self, config, shared_embedding):
        super().__init__()
        self.embed_tokens = shared_embedding
        self.pos_embed = nn.Embedding(config.MAX_SEQ_LEN, config.DIM_MODEL)
        self.embed_dropout = nn.Dropout(config.DROPOUT)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config, is_decoder=True)
             for _ in range(config.NUM_DECODER_LAYERS)]
        )
        self.layer_norm = RMSNorm(config.DIM_MODEL, eps=config.EPS_LAYER_NORM)

    def forward(
        self,
        input_ids,
        encoder_hidden_states,
        self_mask=None,
        cross_mask=None
    ):
        seq_len = input_ids.size(1)
        token_emb = self.embed_tokens(input_ids)
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)
        pos_emb = self.pos_embed(positions)
        hidden_states = token_emb + pos_emb
        hidden_states = self.embed_dropout(hidden_states)

        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                self_mask=self_mask,
                cross_mask=cross_mask,
                encoder_hidden_states=encoder_hidden_states
            )

        return self.layer_norm(hidden_states)


class TransformerConditionalGeneration(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.shared = nn.Embedding(config.VOCAB_SIZE, config.DIM_MODEL)

        self.encoder = TransformerEncoder(config, self.shared)
        self.decoder = TransformerDecoder(config, self.shared)

        self.lm_head = nn.Linear(config.DIM_MODEL, config.VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.shared.weight

    def _shift_right(self, labels):
        shifted = labels.new_zeros(labels.shape)
        shifted[..., 1:] = labels[..., :-1]
        shifted[..., 0] = self.config.BOS_TOKEN_ID
        return shifted

    def forward(
        self,
        input_ids=None,
        labels=None,
        encoder_hidden_states=None,
        decoder_input_ids=None,
        encoder_padding_mask=None,
    ):
        if encoder_padding_mask is None and input_ids is not None:
            encoder_padding_mask = self.make_padding_mask(input_ids)

        if encoder_hidden_states is None:
            encoder_hidden_states = self.encoder(
                input_ids,
                padding_mask=encoder_padding_mask
            )

        if labels is not None:
            decoder_input_ids = self._shift_right(labels)

        if decoder_input_ids is None:
            raise ValueError("Need decoder_input_ids or labels")

        decoder_padding_mask = self.make_padding_mask(decoder_input_ids)
        causal_mask = self.make_causal_mask(
            decoder_input_ids.size(1),
            decoder_input_ids.device
        )

        decoder_hidden_states = self.decoder(
            decoder_input_ids,
            encoder_hidden_states,
            self_mask=decoder_padding_mask + causal_mask,
            cross_mask=encoder_padding_mask
        )

        logits = self.lm_head(decoder_hidden_states)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=self.config.PAD_TOKEN_ID,
                label_smoothing=0.1
            )
        return {"loss": loss, "logits": logits}

    @torch.no_grad()
    def generate(self, input_ids, max_length=40):
        encoder_hidden_states = self.encoder(input_ids)
        encoder_padding_mask = self.make_padding_mask(input_ids)

        decoder_input_ids = torch.full(
            (input_ids.size(0), 1),
            self.config.BOS_TOKEN_ID,
            device=input_ids.device
        )

        for _ in range(max_length):
            outputs = self(
                encoder_hidden_states=encoder_hidden_states,
                decoder_input_ids=decoder_input_ids,
                encoder_padding_mask=encoder_padding_mask
            )

            next_token_logits = outputs["logits"][:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            decoder_input_ids = torch.cat(
                [decoder_input_ids, next_token],
                dim=-1
            )

            if (next_token == self.config.EOS_TOKEN_ID).all():
                break

        return decoder_input_ids  
    def make_padding_mask(self, input_ids):
        if input_ids is None:
            return None
        mask = (input_ids == self.config.PAD_TOKEN_ID)
        mask = mask.unsqueeze(1).unsqueeze(2)  
        return mask.float() * -1e9

    def make_causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.float() * -1e9