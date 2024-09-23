#!/usr/bin/env python3

import torch
from torch import nn
from torch.nn import functional as f

from encoder import Encoder
from decoder import Decoder

class TransformerTranslator(nn.Module):
    def __init__(self,
                 input_vocab_size: int,
                 output_vocab_size: int,
                 embed_dim: int,
                 d_model: int = 512,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 num_attention_heads: int = 4,
                 ffwd_dim: int = 2048,
                 max_seq_len: int = 512,
                 max_output_tokens: int = 1000):
        super(TransformerTranslator, self).__init__()
        self.encoder = Encoder(num_encoder_layers, 
                               input_vocab_size, 
                               embed_dim, 
                               d_model, 
                               max_seq_len, 
                               num_attention_heads, 
                               ffwd_dim)
        self.decoder = Decoder(output_vocab_size,
                               num_decoder_layers,
                               max_output_tokens,
                               num_attention_heads, 
                               embed_dim,
                               d_model,
                               ffwd_dim)
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, targets: torch.Tensor = None) -> torch.Tensor: 
        # x shape (B,T)
        assert len(x.shape) == 2 

        # B,T -> B,T,H
        encoder_out = self.encoder(x)

        # B,T,vocab_size
        decoder_out = self.decoder(targets, encoder_out)
        return decoder_out