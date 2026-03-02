import torch
import sacrebleu
from tqdm import tqdm
from torch.utils.data import DataLoader
from modules.dataset import TranslationDataset, collate_fn, decode_batch, train_sentencepiece
from modules.transformer import TransformerConditionalGeneration
from modules.config import TrainingConfig, ModelConfig
import os
from functools import partial


def train(training_config: TrainingConfig, model, dl_train, dl_val, vocab):
    device = training_config.DEVICE

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.LR
    )

    gradient_accumulation_steps = training_config.GRAD_ACUM
    best_bleu = 0.0
    steps_per_epoch = (len(dl_train) + gradient_accumulation_steps - 1) // gradient_accumulation_steps
    total_optimizer_steps = steps_per_epoch * training_config.NUM_EPOCHS

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=training_config.LR,
        total_steps=total_optimizer_steps,
        pct_start=0.1,              
        anneal_strategy='cos',       
        cycle_momentum=False         
    )
    for epoch in range(training_config.NUM_EPOCHS):
        model.train()
        total_train_loss = 0.0
        pbar = tqdm(dl_train, desc=f"Epoch {epoch+1}", leave=False)
        optimizer.zero_grad()
        for step, (src, tgt) in enumerate(pbar):
            src = src.to(device)
            tgt = tgt.to(device)
            outputs = model(input_ids=src, labels=tgt)
            loss = outputs["loss"] / gradient_accumulation_steps
            loss.backward()
            total_train_loss += loss.item() * gradient_accumulation_steps
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()                   
                optimizer.zero_grad()
            current_lr = scheduler.get_last_lr()[0] 
            pbar.set_postfix(loss=loss.item() * gradient_accumulation_steps,
                             lr=f"{current_lr:.2e}")
        if (step + 1) % gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()                         
            optimizer.zero_grad()

        avg_train_loss = total_train_loss / len(dl_train)
        model.eval()
        total_val_loss = 0.0
        all_preds = []
        all_refs = []

        with torch.no_grad():
            for src, tgt in dl_val:
                src = src.to(device)
                tgt = tgt.to(device)

                outputs = model(input_ids=src, labels=tgt)
                total_val_loss += outputs["loss"].item()

                generated = model.generate(src)

                preds = decode_batch(
                    generated,
                    vocab,
                    pad_id=model.config.PAD_TOKEN_ID,
                    eos_id=model.config.EOS_TOKEN_ID
                )
                refs = decode_batch(
                    tgt,
                    vocab,
                    pad_id=model.config.PAD_TOKEN_ID,
                    eos_id=model.config.EOS_TOKEN_ID
                )
                all_preds.extend(preds)
                all_refs.extend([[r] for r in refs])

        avg_val_loss = total_val_loss / len(dl_val)
        bleu = sacrebleu.corpus_bleu(all_preds, all_refs).score

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | BLEU: {bleu:.2f}")

        if bleu > best_bleu:
            best_bleu = bleu
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/best_model.pt")


if __name__ == "__main__":
    training_config = TrainingConfig()
    model_config = ModelConfig()

    data_folder = training_config.DATA_FOLDER
    train_de = f"{data_folder}/train.de-en.de"
    train_en = f"{data_folder}/train.de-en.en"
    val_de = f"{data_folder}/val.de-en.de"
    val_en = f"{data_folder}/val.de-en.en"
    src_sp = train_sentencepiece([train_de, val_de], "spm_de", vocab_size=model_config.VOCAB_SIZE)
    tgt_sp = train_sentencepiece([train_en, val_en], "spm_en", vocab_size=model_config.VOCAB_SIZE)

    model_config.VOCAB_SIZE = max(src_sp.get_piece_size(), tgt_sp.get_piece_size())
    model_config.PAD_TOKEN_ID = src_sp.pad_id()
    model_config.BOS_TOKEN_ID = src_sp.bos_id()
    model_config.EOS_TOKEN_ID = src_sp.eos_id()

    ds_train = TranslationDataset(
    src_sp, tgt_sp, train_de, train_en,
    train_epoch_len=training_config.TRAIN_EPOCH_LEN
    )
    ds_val = TranslationDataset(
        src_sp, tgt_sp, val_de, val_en
    )
    collate_fn_with_pad = partial(collate_fn, pad_id=model_config.PAD_TOKEN_ID)

    dl_train = DataLoader(
        ds_train,
        batch_size=training_config.BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        collate_fn=collate_fn_with_pad
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=training_config.BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn_with_pad
    )

    model = TransformerConditionalGeneration(model_config).to(training_config.DEVICE)
    train(training_config, model, dl_train, dl_val, tgt_sp)   