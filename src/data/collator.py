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
    def __init__(self, device='cpu'):
        self.aligner = GraphemeAligner().to(device)
        self.featurizer = MelSpectrogram(MelSpectrogramConfig()).to(device)
        self.hop_length = MelSpectrogramConfig().hop_length

    def __call__(self, instances: List[Tuple]) -> Dict:
        waveform, waveforn_length, transcript, tokens, token_lengths = list(
            zip(*instances)
        )
        waveform = pad_sequence([
            waveform_[0] for waveform_ in waveform
        ]).transpose(0, 1)
        waveforn_length = torch.cat(waveforn_length)

        tokens = pad_sequence([
            tokens_[0] for tokens_ in tokens
        ]).transpose(0, 1)
        token_lengths = torch.cat(token_lengths)
        melspec_length = waveforn_length // self.hop_length
        durations = self.aligner(
            waveform, waveforn_length, transcript
        )
        melspec = self.featurizer(waveform)
        token_padded_length = tokens.shape[1]
        melspec_padded_length = melspec.shape[2]
        duration_multipliers = durations * melspec_length[:, None]

        n_mels_for_padds = (melspec_padded_length - melspec_length) / (token_padded_length - token_lengths)
        n_mels_for_padds[n_mels_for_padds.isinf()] = 0
        padding_durations = (torch.arange(token_padded_length)[None, :] > token_lengths[:, None]) * n_mels_for_padds[:, None]
        duration_multipliers += padding_durations
        duration_multipliers = torch.round(duration_multipliers)

        error = melspec_padded_length - duration_multipliers.sum(1)
        error_shift = (torch.arange(token_padded_length)[None, :] < torch.abs(error)[:, None]).int() * torch.sign(error)[:, None]
        duration_multipliers += error_shift
        return {"waveform" : waveform,
                "waveforn_length" : waveforn_length,
                "melspec" : melspec,
                "melspec_length" : melspec_length,
                "transcript" : transcript,
                "tokens" : tokens,
                "token_lengths" : token_lengths,
                "duration_multipliers" : duration_multipliers}