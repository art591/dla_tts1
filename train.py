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

def train(run, epoch, train_dataloader, model, optimizer, scheduler, log_loss_every, log_audio_every):
    model.train()
    i = 0
    for batch in tqdm(train_dataloader):
        pred_mel, pred_len = model(batch)

        mask = (torch.arange(pred_len.shape[1])[None, :].to(device)  <= batch['token_lengths'][:, None]).float()
        loss_len = criterion(pred_len * mask,  torch.log1p(batch["duration_multipliers"]) * mask)

        mask = (torch.arange(pred_mel.shape[1])[None, :].to(device)  <= batch['melspec_length'][:, None]).float()
        loss_mel = criterion(pred_mel * mask[:, :, None], batch['melspec'] * mask[:, :, None])
        loss = loss_mel + loss_len
        if i % log_loss_every == 0 and i != 0:
            run.log({"Train loss" : loss}, step=epoch * len(train_dataloader) + i)
            run.log({"Train Mel Loss" : loss_mel}, step=epoch * len(train_dataloader) + i)
            run.log({"Train Duration Loss" : loss_len}, step=epoch * len(train_dataloader) + i)
            run.log({"Lerarning rate" : optimizer.param_groups[0]['lr']}, step=epoch * len(train_dataloader) + i)
        if i % log_audio_every == 0 and i != 0:
            mel_to_log = pred_mel[0]
            melspec_to_log  = pred_mel[0][:, :batch['melspec_length'][0]].unsqueeze(0)
            reconstructed_wav = vocoder.inference(melspec_to_log).squeeze().detach().cpu().numpy()
            run.log({"Train Audio" : wandb.Audio(reconstructed_wav, 22050)}, step=epoch * len(train_dataloader) + i)
            d = (torch.exp(pred_len[0]) - 1).round().int()
            d[d < 1] = 1
            d1 = d.cumsum(0)
            maxlen = d.sum().item()
            mask1 = torch.arange(maxlen)[None, :].to(device) < (d1[:, None])
            mask2 = torch.arange(maxlen)[None, :].to(device) >= (d1 - d)[:, None]
            mask = (mask2 * mask1).float()
            run.log({"Train Pred durations" : wandb.Image(mask.detach().cpu().numpy())}, step=epoch * len(train_dataloader) + i)
            d = batch['duration_multipliers'][0]
            d1 = d.cumsum(0)
            maxlen = d.sum().item()
            mask1 = torch.arange(maxlen)[None, :].to(device) < (d1[:, None])
            mask2 = torch.arange(maxlen)[None, :].to(device) >= (d1 - d)[:, None]
            mask = (mask2 * mask1).float()
            run.log({"Train True durations" : wandb.Image(mask.detach().cpu().numpy())}, step=epoch * len(train_dataloader) + i)
        optimizer.zero_grad()
        loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        scheduler.step()
        i += 1


def validation(run, iteration, model):
    model.eval()
    tokenizer  = torchaudio.pipelines.TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.get_text_processor()
    sentences = ['A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest',
                 'Massachusetts Institute of Technology may be best known for its math, science and engineering education',
                 'Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space']
    print("Start Validation")
    audios = []
    durations = []
    for k, sentence in enumerate(sentences):
        tokens, length = tokenizer(sentence.lower())
        tokens = tokens.to(device)
        length = length.to(device)
        batch = {}
        batch['tokens'] = tokens
        batch['token_lengths'] = length
        pred_mel, pred_len = model(batch, False)
        melspec_to_log  = pred_mel
        reconstructed_wav = vocoder.inference(melspec_to_log).squeeze().detach().cpu().numpy()
        d = (torch.exp(pred_len[0]) - 1).round().int()
        d[d < 1] = 1
        d1 = d.cumsum(0)
        maxlen = d.sum().item()
        mask1 = torch.arange(maxlen)[None, :].to(device) < (d1[:, None])
        mask2 = torch.arange(maxlen)[None, :].to(device) >= (d1 - d)[:, None]
        mask = (mask2 * mask1).float()
        audios.append(wandb.Audio(reconstructed_wav, 22050, caption=sentence))
        durations.append(wandb.Image(mask.detach().cpu().numpy(), caption=sentence))
    run.log({"Val Audio" : audios}, step=iteration)
    run.log({"Val durations" : durations}, step=iteration)


if __name__ == '__main__':
    experiment_path = 'real_alignments'
    project_name = 'tts_1'
    name = 'real_alignments'
    log_audio_every = 100
    log_loss_every = 5
    save_every = 10
    n_epochs = 20
    batch_size = 16
    device = 'cuda'
    train_dataloader = DataLoader(LJSpeechDataset('/home/jupyter/mnt/datasets/LJSpeech/'), batch_size=batch_size, collate_fn=LJSpeechCollator(device, aligner_mode='real'), shuffle=True)
    model = FastSpeechModel(38, 10000, 6, 80, 256, 1024, (9, 1), 'gelu', 2, 128, 'pre', 0.1, device).to(device)
    vocoder = Vocoder().eval().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
                                              1e-4,
                                              total_steps=n_epochs * len(train_dataloader),
                                              div_factor=1e+4,
                                              pct_start=0.05,
                                              anneal_strategy='linear')
    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)
    with wandb.init(project=project_name, name=name) as run:
        for i in range(n_epochs):
            print(f'Start Epoch {i}')
            train(run, i, train_dataloader, model, optimizer, scheduler, log_loss_every, log_audio_every)
            if i % save_every == 0:
                torch.save(model.state_dict(), f"{experiment_path}/model.pth")
                torch.save(optimizer.state_dict(), f"{experiment_path}/optimizer.pth")
            validation(run, (i + 1) * len(train_dataloader), model)
