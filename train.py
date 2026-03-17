
import torch
from torch.utils.data import DataLoader
from functools import partial
from tqdm import tqdm
from modules.config import TrainConfig, ModelConfig
from modules.transformer_second_impl import TransformerMT
from modules.dataset import train_spm, TranslationDataset, collate
import sacrebleu
import wandb
from torch.cuda.amp import autocast, GradScaler


def train_model(config, train_loader, val_loader, model, src_sp, tgt_sp, val_ref_file):
    device = config.DEVICE
    model.to(device)
    if config.COMPILE and hasattr(torch, 'compile'):
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR)
    steps_per_epoch = len(train_loader) // config.GRAD_ACCUM_STEPS
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, config.LR,
        epochs=config.NUM_EPOCHS,
        steps_per_epoch=steps_per_epoch
    )

    scaler = GradScaler('cuda', enabled=config.USE_BF16)
    best_bleu = 0.0

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        total_train_loss = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        for step, (src, tgt) in enumerate(pbar, 1):
            src, tgt = src.to(device), tgt.to(device)

            if config.USE_BF16:
                with autocast('cuda', dtype=torch.bfloat16):
                    loss, _ = model(src, labels=tgt)
            else:
                loss, _ = model(src, labels=tgt)

            scaled_loss = loss / config.GRAD_ACCUM_STEPS
            scaler.scale(scaled_loss).backward()

            if step % config.GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            total_train_loss += loss.item()
            pbar.set_postfix(loss=loss.item(), step=step)

        model.eval()
        total_val_loss = 0
        preds = []
        with torch.no_grad():
            for src, tgt in tqdm(val_loader, desc='Validating', leave=False):
                src, tgt = src.to(device), tgt.to(device)

                if config.USE_BF16:
                    with autocast('cuda', dtype=torch.bfloat16):
                        loss, _ = model(src, labels=tgt)
                else:
                    loss, _ = model(src, labels=tgt)

                total_val_loss += loss.item()
                out = model.generate(src, max_len=100)
                for seq in out:
                    ids = [i for i in seq.cpu().tolist()
                           if i not in (model.config.PAD_TOKEN_ID,
                                        model.config.EOS_TOKEN_ID,
                                        model.config.BOS_TOKEN_ID)]
                    preds.append(tgt_sp.decode(ids))

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        bleu = sacrebleu.corpus_bleu(preds,
            [[line.strip() for line in open(val_ref_file, encoding='utf-8')]]).score

        print(f'Epoch {epoch+1}: train loss {avg_train_loss:.4f}, '
              f'val loss {avg_val_loss:.4f}, BLEU {bleu:.2f}')

        if bleu > best_bleu:
            best_bleu = bleu
            state_dict = model.state_dict()
            if any(k.startswith('_orig_mod.') for k in state_dict):
                new_state_dict = {k.replace('_orig_mod.', ''): v
                                  for k, v in state_dict.items()}
            else:
                new_state_dict = state_dict
            torch.save(new_state_dict, config.SAVE_PATH)

        if config.LOG_WANDB:
            wandb.log({'train_loss': avg_train_loss,
                       'val_loss': avg_val_loss,
                       'bleu': bleu})
if __name__=="__main__":
    train_cfg = TrainConfig()
    model_cfg = ModelConfig()
    # os.makedirs(os.path.dirname(train_cfg.SAVE_PATH), exist_ok=True)
    src_sp = train_spm([f"{train_cfg.DATA_FOLDER}/train.de-en.de",
                        f"{train_cfg.DATA_FOLDER}/val.de-en.de"], 'spm_de', model_cfg.VOCAB_SIZE)
    tgt_sp = train_spm([f"{train_cfg.DATA_FOLDER}/train.de-en.en",
                        f"{train_cfg.DATA_FOLDER}/val.de-en.en"], 'spm_en', model_cfg.VOCAB_SIZE)

    model_cfg.VOCAB_SIZE = max(src_sp.vocab_size(), tgt_sp.vocab_size())
    model_cfg.PAD_TOKEN_ID = src_sp.pad_id()
    model_cfg.BOS_TOKEN_ID = src_sp.bos_id()
    model_cfg.EOS_TOKEN_ID = src_sp.eos_id()

    train_ds = TranslationDataset(src_sp, tgt_sp,
                                    f"{train_cfg.DATA_FOLDER}/train.de-en.de",
                                    f"{train_cfg.DATA_FOLDER}/train.de-en.en")
    val_ds = TranslationDataset(src_sp, tgt_sp,
                                f"{train_cfg.DATA_FOLDER}/val.de-en.de",
                                f"{train_cfg.DATA_FOLDER}/val.de-en.en")

    collate_fn = partial(collate, pad_id=model_cfg.PAD_TOKEN_ID)
    train_loader = DataLoader(train_ds, batch_size=train_cfg.BATCH_SIZE, shuffle=True,
                            collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=train_cfg.BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn, pin_memory=True)

    model = TransformerMT(model_cfg)

    if train_cfg.LOG_WANDB:
        wandb.init(project='translation-minimal', config={**train_cfg.__dict__, **model_cfg.__dict__})

    train_model(train_cfg, train_loader, val_loader, model, src_sp, tgt_sp,
                f"{train_cfg.DATA_FOLDER}/val.de-en.en")

    if train_cfg.LOG_WANDB:
        wandb.finish()
