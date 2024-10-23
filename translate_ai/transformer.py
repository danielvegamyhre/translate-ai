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
                 input_seq_len: int = 512,
                 output_seq_len: int = 1000,
                 device: str = "cpu"):
        
        super(TransformerTranslator, self).__init__()

        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.embed_dim = embed_dim
        self.d_model = d_model
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_attention_heads = num_attention_heads
        self.ffwd_dim = ffwd_dim
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len 
        self.device = device
        
        self.encoder = Encoder(num_encoder_layers, 
                               input_vocab_size, 
                               embed_dim=embed_dim, 
                               d_model=d_model, 
                               num_attention_heads=num_attention_heads, 
                               ffwd_dim=ffwd_dim,
                               input_seq_len=input_seq_len,
                               device=device)
        
        self.decoder = Decoder(output_vocab_size,
                               num_layers=num_decoder_layers,
                               num_heads=num_attention_heads, 
                               embed_dim=embed_dim,
                               d_model=d_model,
                               ffwd_dim=ffwd_dim,
                               output_seq_len=output_seq_len,
                               device=device)
        
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
                decoder_padding_mask: torch.Tensor = None) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # x shape (B,T)
        assert len(encoder_input.shape) == 2 

        # B,T -> B,T,H
        encoder_out, encoder_sparsity_loss = self.encoder(encoder_input, encoder_padding_mask)

        # B,T,vocab_size
        decoder_out, decoder_sparsity_loss = self.decoder(decoder_input, encoder_out, decoder_padding_mask)

        sparsity_loss = encoder_sparsity_loss + decoder_sparsity_loss
        return decoder_out, sparsity_loss