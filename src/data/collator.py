from typing import Tuple, Dict, Optional, List, Union
from itertools import islice
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass
from src.data.aligner import GraphemeAligner
from src.data.melspectrogram import MelSpectrogram, MelSpectrogramConfig


@dataclass
class Batch:
    waveform: torch.Tensor
    waveforn_length: torch.Tensor
    melspec: torch.Tensor
    melspec_length: torch.Tensor
    transcript: List[str]
    tokens: torch.Tensor
    token_lengths: torch.Tensor
    duration_multipliers: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> 'Batch':
        raise NotImplementedError


class LJSpeechCollator:
    def __init__(self, device='cpu', aligner_mode='sasha'):
        self.device = device
        self.aligner_mode = aligner_mode
        if aligner_mode == 'sasha':
            self.aligner = GraphemeAligner().to(device)
        self.featurizer = MelSpectrogram(MelSpectrogramConfig()).to(device)
        self.hop_length = MelSpectrogramConfig().hop_length

    def __call__(self, instances: List[Tuple]) -> Dict:
        waveform, waveforn_length, transcript, tokens, token_lengths, fp_duration_multiplayers = list(
            zip(*instances)
        )
        waveform = pad_sequence([
            waveform_[0] for waveform_ in waveform
        ]).transpose(0, 1).to(self.device)
        waveforn_length = torch.cat(waveforn_length)

        tokens = pad_sequence([
            tokens_[0] for tokens_ in tokens
        ]).transpose(0, 1).to(self.device)
        token_lengths = torch.cat(token_lengths)
        if self.aligner_mode == 'sasha':
            melspec_length = waveforn_length // self.hop_length + 1
            durations = self.aligner(
                waveform, waveforn_length, transcript
            )
            durations = durations / durations.sum(1)[:, None]
            melspec = self.featurizer(waveform)
            token_padded_length = tokens.shape[1]
            durations = durations[:, :token_padded_length]
            melspec_padded_length = melspec.shape[2]
            duration_multipliers = durations * melspec_length[:, None]
            duration_multipliers = duration_multipliers.round().int()


            error = melspec_length - duration_multipliers.sum(1)
            error_shift = (torch.arange(token_padded_length)[None, :] < torch.abs(error)[:, None]).int() * torch.sign(error)[:, None]
            duration_multipliers += error_shift
        else:
            duration_multipliers = fp_duration_multiplayers
            duration_multipliers = pad_sequence([dur for dur in duration_multipliers]).transpose(0, 1).to(self.device)
            melspec_length = waveforn_length // self.hop_length + 1
            melspec = self.featurizer(waveform)
            duration_multipliers = duration_multipliers[:, :tokens.shape[1]]
            tokens = tokens[:, :duration_multipliers.shape[1]]
            melspec = melspec[:, :, :duration_multipliers.sum(1).max()]
            
        return {"waveform" : waveform,
                "waveforn_length" : waveforn_length.to(self.device),
                "melspec" : melspec,
                "melspec_length" : melspec_length.to(self.device),
                "transcript" : transcript,
                "tokens" : tokens,
                "token_lengths" : token_lengths.to(self.device),
                "duration_multipliers" : duration_multipliers.to(self.device)}