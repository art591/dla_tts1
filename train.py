import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchaudio
from src.data.dataset import LJSpeechDataset
from src.data.collator import LJSpeechCollator
from src.models.fastspeech_model import FastSpeechModel
from vocoder import Vocoder
import wandb
from tqdm import tqdm

def train(run, train_dataloader, model, optimizer, scheduler, log_loss_every, log_audio_every):
    model.train()
    for batch in tqdm(train_dataloader):
        pred_mel, pred_len = model(batch)

        mask = (torch.arange(pred_len.shape[1])[None, :] <= batch['token_lengths'][:, None]).float()
        loss_len = criterion(pred_len * mask,  torch.log1p(batch["duration_multipliers"]) * mask)

        mask = (torch.arange(pred_mel.shape[1])[None, :] <= batch['melspec_length'][:, None]).float()
        loss_mel = criterion(pred_mel * mask[:, :, None], batch['melspec'] * mask[:, :, None])
        loss = loss_mel + loss_len
        if i % log_loss_every == 0 and i != 0:
            run.log({"Train loss" : loss})
        if i % log_audio_every == 0 and i != 0:
            print("Train audio")
            mel_to_log = pred_mel[0]
            melspec_to_log  = pred_mel[0][:, :batch['melspec_length'][0]].unsqueeze(0)
            reconstructed_wav = vocoder.inference(melspec_to_log).squeeze().detach().cpu().numpy()
            run.log({"Train Audio" : wandb.Audio(reconstructed_wav, 22050)})
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        scheduler.step()
        break


def validation(run, model):
    model.eval()
    tokenizer  = torchaudio.pipelines.TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.get_text_processor()
    sentences = ['A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest',
                 'Massachusetts Institute of Technology may be best known for its math, science and engineering education',
                 'Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space']
    print("Start Validation")
    for sentence in sentences:
        tokens, length = tokenizer(sentence)
        batch = {}
        batch['tokens'] = tokens
        batch['token_lengths'] = length
        pred_mel, _ = model(batch, False)
        melspec_to_log  = pred_mel
        reconstructed_wav = vocoder.inference(melspec_to_log).squeeze().detach().cpu().numpy()
        run.log({"Val Audio" : wandb.Audio(reconstructed_wav, 22050)})


if __name__ == '__main__':
    experiment_path = 'first_try'
    project_name = 'tts_1'
    name = 'datasphere_first'
    log_audio_every = 250
    log_loss_every = 10
    save_every = 10
    n_epochs = 20
    batch_size = 64
    train_dataloader = DataLoader(LJSpeechDataset('.'), batch_size=batch_size, collate_fn=LJSpeechCollator())
    model = FastSpeechModel(38, 10000, 6, 80, 256, 1024, (9, 1), 'gelu', 2, 128, 'pre', 0.1)
    vocoder = Vocoder().eval()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
                                              1e-3,
                                              total_steps=n_epochs * len(train_dataloader),
                                              div_factor=1000,
                                              pct_start=0.2)
    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)
    with wandb.init(project=project_name, name=name) as run:
        for i in range(n_epochs):
            print(f"Start Epoch {i}")
            train(run, train_dataloader, model, optimizer, scheduler, log_loss_every, log_audio_every)
            if i % save_every == 0:
                torch.save(model.state_dict(), f"{experiment_path}/model.pth")
                torch.save(optimizer.state_dict(), f"{experiment_path}/optimizer.pth")
            validation(run, model)
