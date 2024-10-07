#!/usr/bin/env python3
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from typing import Protocol

class TokenizerInterface(Protocol):
    def encode(self, text: str) -> list[int]:
        pass

class EnglishToSpanishDataset(Dataset):
    def __init__(self, 
                 file_path: str, 
                 tokenizer: TokenizerInterface, 
                 padding_token: int,
                 bos_token: int,
                 eos_token: int):
        
        df = pd.read_csv(file_path)
        
        english_tokenized_lines = [
            torch.tensor([bos_token] + tokenizer.encode(line.strip()) + [eos_token])
            for line in df['english'] 
        ]

        spanish_tokenized_lines = [
            torch.tensor([bos_token] + tokenizer.encode(line.strip()) + [eos_token])
            for line in df['spanish']
        ]

        # pad sequences so they all have the same length
        self.X = torch.stack([
            pad_sequence(english_tokenized_lines, batch_first=True, padding_value=padding_token)
        ]).squeeze()

        self.Y = torch.stack([
            pad_sequence(spanish_tokenized_lines, batch_first=True, padding_value=padding_token)
        ]).squeeze()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx]
