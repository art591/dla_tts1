import torch
from torch import nn
import torchaudio
import numpy as np


class LJSpeechDataset(torchaudio.datasets.LJSPEECH):

    def __init__(self, root):
        super().__init__(root=root)
        self._tokenizer = torchaudio.pipelines.TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.get_text_processor()

    def __getitem__(self, index: int):
        waveform, _, _, transcript = super().__getitem__(index)
        duration_multiplayers = torch.from_numpy(np.load(f'alignments/{index}.npy'))
        waveforn_length = torch.tensor([waveform.shape[-1]]).int()
        transcript = transcript.lower()
        transcript = transcript.replace("mr.", "mister")
        transcript = transcript.replace("ms.", "miss")
        transcript = transcript.replace("mrs.", "misses")
        tokens, token_lengths = self._tokenizer(transcript)

        return waveform, waveforn_length, transcript, tokens, token_lengths, duration_multiplayers

    def decode(self, tokens, lengths):
        result = []
        for tokens_, length in zip(tokens, lengths):
            text = "".join([
                self._tokenizer.tokens[token]
                for token in tokens_[:length]
            ])
            result.append(text)
        return result