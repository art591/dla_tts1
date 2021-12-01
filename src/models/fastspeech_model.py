import torch
from torch import nn
from src.models.transformer_block import TransformerEncoderLayer
import math


class Transformer(nn.Module):
    def __init__(self, n_layers,
                       model_size,
                       intermidiate_size,
                       itermidiate_kernel_size,
                       activation,
                       n_heads,
                       size_per_head,
                       normalization_type,
                       dropout_prob):
        super().__init__()
        self.n_layers = n_layers
        args = [model_size,
                intermidiate_size,
                itermidiate_kernel_size,
                activation,
                n_heads,
                size_per_head,
                normalization_type,
                dropout_prob]
        self.layers = nn.ModuleList([TransformerEncoderLayer(*args) for i in range(n_layers)])

    def forward(self, x, attention_mask):
        for i in range(self.n_layers):
            x, attention_mask = self.layers[i](x, attention_mask)
        return x


class SinCosPositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size,
                 maxlen):
        super().__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, embeddings):
        return embeddings + self.pos_embedding[:embeddings.shape[0], :]


def duplicate_by_duration(encoder_result, durations):
    bs = encoder_result.shape[0]
    encoder_result = encoder_result.reshape(-1, encoder_result.shape[2])
    melspec_len = durations[0].sum()
    assert(torch.all(durations.sum(1) == melspec_len))
    durations = durations.flatten()
    durations_cumsum = durations.cumsum(0)
    mask1 = torch.arange(bs * melspec_len)[None, :] < (durations_cumsum[:, None])
    mask2 = torch.arange(bs * melspec_len)[None, :] >= (durations_cumsum - durations)[:, None]
    mask = (mask2 * mask1).float()
    encoder_result = mask.T @ encoder_result
    encoder_result = encoder_result.reshape(bs, int(melspec_len.item()), encoder_result.shape[1])
    return encoder_result


class FastSpeechModel(nn.Module):
    def __init__(self, vocab_size,
                       max_len,
                       n_layers,
                       output_size,
                       model_size,
                       intermidiate_size,
                       itermidiate_kernel_size,
                       activation,
                       n_heads,
                       size_per_head,
                       normalization_type,
                       dropout_prob):
        super().__init__()
        if activation == 'relu':
            activation = nn.ReLU
        args = [n_layers,
                model_size,
                intermidiate_size,
                itermidiate_kernel_size,
                activation,
                n_heads,
                size_per_head,
                normalization_type,
                dropout_prob]
        self.tokens_positions = SinCosPositionalEncoding(model_size, max_len)
        self.frames_positions = SinCosPositionalEncoding(model_size, max_len)
        self.embedding_layer = nn.Embedding(vocab_size, model_size)
        self.encoder = Transformer(*args)
        self.decoder = Transformer(*args)
        self.output_layer = nn.Linear(model_size, output_size)

    def forward(self, batch):
        tokens = batch["tokens"]
        melspec = batch["melspec"]
        melspec_length = batch["melspec_length"]
        tokens_length = batch["token_lengths"]
        duration_multipliers = batch["duration_multipliers"]

        tokens_embeddings = self.embedding_layer(tokens)
        tokens_embeddings = tokens_embeddings + self.tokens_positions(tokens_embeddings)

        attention_mask = (torch.arange(tokens.shape[1])[None, :] > tokens_length[:, None]).float()
        attention_mask[attention_mask == 1] = -torch.inf
        encoder_result = self.encoder(tokens_embeddings, attention_mask)

        input_to_decoder = duplicate_by_duration(encoder_result, duration_multipliers)
        mask = (torch.arange(input_to_decoder.shape[1])[None, :] <= melspec_length[:, None]).float()
        input_to_decoder = input_to_decoder * mask[:, :, None]
        attention_mask = (torch.arange(input_to_decoder.shape[1])[None, :] > melspec_length[:, None]).float()
        attention_mask[attention_mask == 1] = -torch.inf
        output = self.decoder(input_to_decoder, attention_mask)

        output = self.output_layer(output)
        output = output.permute(0, 2, 1)
        return output
        # output_attention_mask = (torch.arange(input_to_decoder.shape[1])[None, :] <= melspec_length[:, None]).float()
        # return output * output_attention_mask[:, None, :]
