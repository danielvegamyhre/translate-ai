#!/usr/bin/env python3

import torch
from torch import nn
from torch.nn import functional as f
from torch.utils.data import Dataset
from dataclasses import dataclass
from argparse import ArgumentParser
import tiktoken
from tqdm import tqdm

from transformer import TransformerTranslator
from checkpoint import load_checkpoint
from train import TrainingConfig

def translate(english_query: str, 
              checkpoint_file: str, 
              pad_token: int = 0, 
              bos_token: int = 1, 
              eos_token: int = 2) -> str:
    '''Translate English input sequence to Spanish.'''
    
    tokenizer = tiktoken.get_encoding("cl100k_base")
    print(f"tokenizer: tiktoken cl100k_base")
    input_tokens = torch.tensor(tokenizer.encode(english_query)).unsqueeze(0) # (B,T) where B=1

    # load checkpoint if specified
    cfg = None
    if checkpoint_file:
        checkpoint = load_checkpoint(checkpoint_file)
        cfg = checkpoint['config']

    model = TransformerTranslator(
        input_vocab_size=tokenizer.n_vocab, 
        output_vocab_size=tokenizer.n_vocab,
        embed_dim=512,
        d_model=512,
        num_encoder_layers=cfg.num_layers if cfg else 6,
        num_decoder_layers=cfg.num_layers if cfg else 6,
        num_attention_heads=8,
        ffwd_dim=2048,
        max_seq_len=128,
        max_output_tokens=128
    )

    if cfg:
        model = model.to(cfg.device)
        input_tokens = input_tokens.to(cfg.device)

    # run input query through encoder to get encoder output / context
    encoder_out = model.encoder(input_tokens)

    # run decoder one step at time auto-regressively
    # 0 is the padding token
    with torch.no_grad():
        pred_tokens = torch.tensor([0], device=cfg.device).unsqueeze(0) # (B,1) where B=1
        for _ in tqdm(range(model.max_output_tokens)):
            decoder_out = model.decoder(pred_tokens, encoder_out)       # (B,T,output_vocab_size) where B=1
            # get latest predicted token in seq
            decoder_out = decoder_out[:, -1, :]                         # (B,1,output_vocab_size) where B=1
            next_token = torch.argmax(decoder_out, dim=-1).unsqueeze(0) # (B,T) where B=1 and T=1
            pred_tokens = torch.cat([pred_tokens, next_token], dim=-1)  # (B,T) where B=1 and T=T+1
            # if next token is padding token, end translation
            if next_token.item() == padding_token:
                break
    
    # decoder predicted tokens into spanish
    decoded = tokenizer.decode(pred_tokens.squeeze().tolist())
    return decoded

if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument("--english-query", type=str, required=True)
    argparser.add_argument("--checkpoint-file", type=str, required=True)
    args = argparser.parse_args()
    print(translate(args.english_query, args.checkpoint_file))
