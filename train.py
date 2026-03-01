import torch
import sacrebleu
from tqdm import tqdm
from torch.utils.data import DataLoader
from modules.dataset import build_vocab_from_files, TranslationDataset, collate_fn, decode_batch
from modules.transformer import TransformerConditionalGeneration
from modules.config import TrainingConfig, ModelConfig

def train(training_config:TrainingConfig, model, dl_train, dl_val, vocab):

    device = training_config.DEVICE

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.LR
    )

    gradient_accumulation_steps = training_config.GRAD_ACUM   
    best_bleu = 0.0

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
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            pbar.set_postfix(loss=loss.item() * gradient_accumulation_steps)

        if (step + 1) % gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
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

        bleu = sacrebleu.corpus_bleu(
            all_preds,
            list(zip(*all_refs))
        ).score

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss  : {avg_val_loss:.4f}")
        print(f"BLEU      : {bleu:.2f}\n")

        if bleu > best_bleu:
            best_bleu = bleu
            torch.save(model.state_dict(), "models/best_model.pt")


if __name__ == "__main__":

    training_config = TrainingConfig()
    model_config = ModelConfig()
    vocab = build_vocab_from_files([
        f"{training_config.DATA_FOLDER}/train.de-en.de",
        f"{training_config.DATA_FOLDER}/train.de-en.en",
        f"{training_config.DATA_FOLDER}/val.de-en.de",
        f"{training_config.DATA_FOLDER}/val.de-en.en",
    ])
    print(f"VOCAB SIZE is: {len(vocab)}")
    model_config.VOCAB_SIZE = len(vocab)

    model_config.PAD_TOKEN_ID = vocab["<pad>"]
    model_config.BOS_TOKEN_ID = vocab["<bos>"]
    model_config.EOS_TOKEN_ID = vocab["<eos>"]
    ds_train = TranslationDataset(
        vocab,
        vocab,
        f"{training_config.DATA_FOLDER}/train.de-en.de",
        f"{training_config.DATA_FOLDER}/train.de-en.en",
        train_epoch_len=training_config.TRAIN_EPOCH_LEN
        
    )

    ds_val = TranslationDataset(
        vocab,
        vocab,
        f"{training_config.DATA_FOLDER}/val.de-en.de",
        f"{training_config.DATA_FOLDER}/val.de-en.en"
    )

    dl_train = DataLoader(
        ds_train,
        batch_size=training_config.BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        collate_fn=collate_fn
    )

    dl_val = DataLoader(
        ds_val,
        batch_size=training_config.BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn
    )
    model = TransformerConditionalGeneration(model_config)
    model = model.to(training_config.DEVICE)
    train(training_config, model, dl_train, dl_val, vocab)