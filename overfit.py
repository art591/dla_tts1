import os
from torch import nn, optim
from torch.utils.data import DataLoader
from catalyst import dl, utils
from src.data.dataset import LJSpeechDataset
from src.data.collator import LJSpeechCollator
from src.models.fastspeech_model import FastSpeechModel
from vocoder import Vocoder
import wandb
from tqdm import tqdm

dataloader = DataLoader(LJSpeechDataset('.'), batch_size=10, collate_fn=LJSpeechCollator())
model = FastSpeechModel(51, 10000, 1, 80, 30, 60, 9, nn.ReLU, 1, 128, 'post', 0)
vocoder = Vocoder().eval()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

log_audio_every = 250
log_loss_every = 10

with wandb.init(project="tts_1", name="overfit") as run:
    batch = next(iter(dataloader))
    for i in tqdm(range(1, 5000)):
        result = model(batch)
        loss = criterion(result, batch['melspec'])
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
