import torch
import sacrebleu
from tqdm import tqdm
from torch.utils.data import DataLoader
from modules.dataset import build_tokenizer, TranslationDataset, collate_fn, decode_batch
from modules.transformer import TransformerConditionalGeneration
from modules.config import TrainingConfig, ModelConfig

def train(training_config: TrainingConfig, model, dl_train, dl_val, vocab):
    device = training_config.DEVICE

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.LR
    )

    gradient_accumulation_steps = training_config.GRAD_ACUM
    best_bleu = 0.0

    total_batches = len(dl_train)
    total_optimizer_steps = (total_batches * training_config.NUM_EPOCHS 
                             + gradient_accumulation_steps - 1) // gradient_accumulation_steps
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
        for step, (src, tgt, _, _) in enumerate(pbar):
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
            for src, tgt, _, _ in dl_val:
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
        bleu = sacrebleu.corpus_bleu(all_preds, list(zip(*all_refs))).score

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | BLEU: {bleu:.2f}")

        if bleu > best_bleu:
            best_bleu = bleu
            torch.save(model.state_dict(), "models/best_model.pt")


if __name__ == "__main__":
    training_config = TrainingConfig()
    model_config = ModelConfig()
    data_folder = training_config.DATA_FOLDER
    train_de = f"{data_folder}/train.de-en.de"
    train_en = f"{data_folder}/train.de-en.en"
    val_de = f"{data_folder}/val.de-en.de"
    val_en = f"{data_folder}/val.de-en.en"
    src_tokenizer = build_tokenizer([train_de, val_de], spacy_lang="de_core_news_sm", vocab_size=training_config.VOCAB_SIZE)
    tgt_tokenizer = build_tokenizer([train_en, val_en], spacy_lang="en_core_web_sm", vocab_size=training_config.VOCAB_SIZE)
    model_config.VOCAB_SIZE = max(src_tokenizer.get_vocab_size(), tgt_tokenizer.get_vocab_size())
    model_config.PAD_TOKEN_ID = src_tokenizer.token_to_id("<pad>")
    model_config.BOS_TOKEN_ID = src_tokenizer.token_to_id("<bos>")
    model_config.EOS_TOKEN_ID = src_tokenizer.token_to_id("<eos>")
    ds_train = TranslationDataset(
        src_tokenizer, tgt_tokenizer,
        train_de, train_en,
        train_epoch_len=training_config.TRAIN_EPOCH_LEN
    )
    ds_val = TranslationDataset(
        src_tokenizer, tgt_tokenizer,
        val_de, val_en
    )
    dl_train = DataLoader(ds_train, batch_size=training_config.BATCH_SIZE, shuffle=True,
                          pin_memory=True, collate_fn=collate_fn)
    dl_val = DataLoader(ds_val, batch_size=training_config.BATCH_SIZE, shuffle=False,
                        pin_memory=True, collate_fn=collate_fn)
    model = TransformerConditionalGeneration(model_config).to(training_config.DEVICE)
    train(training_config, model, dl_train, dl_val, tgt_tokenizer)