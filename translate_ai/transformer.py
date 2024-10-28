#!/usr/bin/env python3

import torch
from torch import nn

from encoder import Encoder
from decoder import Decoder

class DifferentialTransformer(nn.Module):
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
        
        super(DifferentialTransformer, self).__init__()

        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.embed_dim = embed_dim
        self.d_model = d_model
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_attention_heads = num_attention_heads
        self.ffwd_dim = ffwd_dim
        self.max_seq_len = max_seq_len
        self.max_output_tokens = max_output_tokens 
        
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
                               ffwd_dim,
                               max_seq_len)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.xavier_normal_(module.weight)

    def forward(self, 
                encoder_input: torch.Tensor, 
                decoder_input: torch.Tensor,
                encoder_padding_mask: torch.Tensor = None,
                decoder_padding_mask: torch.Tensor = None) -> torch.Tensor: 
        # x shape (B,T)
        assert len(encoder_input.shape) == 2 

        # B,T -> B,T,H
        encoder_out = self.encoder(encoder_input, encoder_padding_mask)

        # B,T,vocab_size
        decoder_out = self.decoder(decoder_input, encoder_out, decoder_padding_mask)
        return decoder_out