import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from src.data.dataset import LJSpeechDataset
from src.data.collator import LJSpeechCollator
from src.models.fastspeech_model import FastSpeechModel
from vocoder import Vocoder
import wandb
from tqdm import tqdm

dataloader = DataLoader(LJSpeechDataset('.'), batch_size=10, collate_fn=LJSpeechCollator())
model = FastSpeechModel(51, 10000, 1, 80, 30, 60, (9, 1), 'gelu', 1, 128, 'post', 0)
vocoder = Vocoder().eval()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

log_audio_every = 250
log_loss_every = 10

with wandb.init(project="tts_1", name="overfit_2") as run:
    batch = next(iter(dataloader))
    for i in tqdm(range(1, 5000)):
        pred_mel, pred_len = model(batch)

        mask = (torch.arange(pred_len.shape[1])[None, :] <= batch['token_lengths'][:, None]).float()
        loss_len = criterion(pred_len * mask,  torch.log1p(batch["duration_multipliers"]) * mask)

        mask = (torch.arange(pred_mel.shape[1])[None, :] <= batch['melspec_length'][:, None]).float()
        loss_mel = criterion(pred_mel * mask[:, :, None], batch['melspec'] * mask[:, :, None])
        loss = loss_mel + loss_len
        if i % log_loss_every == 0:
            run.log({"loss" : loss})
        if i % log_audio_every == 0:
            print("Logging audio")
            mel_to_log = result[0]
            melspec_to_log  = result[0][:, :batch['melspec_length'][0]].unsqueeze(0)
            reconstructed_wav = vocoder.inference(melspec_to_log).squeeze().detach().cpu().numpy()
            run.log({"Audio" : wandb.Audio(reconstructed_wav, 22050)})
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
