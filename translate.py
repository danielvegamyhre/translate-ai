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

def translate(english_query: str, checkpoint_file: str) -> str:
    '''Translate English input sequence to Spanish.'''

    # load checkpoint if specified
    checkpoint = load_checkpoint(checkpoint_file)
    cfg = checkpoint['config']

    # configure tokenizer
    tokenizer = tiktoken.get_encoding("cl100k_base")
    vocab_size = tokenizer.n_vocab + 3 # +3 for BOS, EOS, PAD tokens
    pad_token = vocab_size - 1
    bos_token = vocab_size - 2
    eos_token = vocab_size - 3
    print(f"tokenizer: tiktoken cl100k_base")

    # (B,T) where B=1
    input_tokens = torch.tensor([bos_token] + tokenizer.encode(english_query) + [eos_token]).unsqueeze(0).to(cfg.device) 

    model = TransformerTranslator(
        input_vocab_size=vocab_size, 
        output_vocab_size=vocab_size,
        embed_dim=512,
        d_model=512,
        num_encoder_layers=cfg.num_layers,
        num_decoder_layers=cfg.num_layers,
        num_attention_heads=8,
        ffwd_dim=2048,
        max_seq_len=128,
        max_output_tokens=128
    ).to(cfg.device)

    model.load_state_dict(checkpoint['model_state_dict'])

    # run input query through encoder to get encoder output / context
    encoder_out = model.encoder(input_tokens)

    # run decoder one step at time auto-regressively
    with torch.no_grad():
        pred_tokens = torch.tensor([bos_token], device=cfg.device).unsqueeze(0) # (B,1) where B=1
        for _ in tqdm(range(model.max_output_tokens)):
            
            decoder_out = model.decoder(pred_tokens, encoder_out)               # (B,T,output_vocab_size) where B=1

            # get latest predicted token in seq
            decoder_out = decoder_out[:, -1, :]                                 # (B,1,output_vocab_size) where B=1
            next_token_probs = torch.softmax(decoder_out, dim=-1)
            next_token = torch.argmax(next_token_probs, dim=-1).unsqueeze(0)    # (B,T) where B=1 and T=1
            pred_tokens = torch.cat([pred_tokens, next_token], dim=-1)          # (B,T) where B=1 and T=T+1

            # if next token is EOS token, end translation
            if next_token.item() == eos_token:
                break
    
    # decoder predicted tokens into spanish
    cleaned_pred_tokens = [token for token in pred_tokens.squeeze().tolist() if token not in {bos_token, eos_token, pad_token}]
    decoded = tokenizer.decode(cleaned_pred_tokens)
    return decoded

if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument("--english-query", type=str, required=True)
    argparser.add_argument("--checkpoint-file", type=str, required=True)
    args = argparser.parse_args()
    print(translate(args.english_query, args.checkpoint_file))
