#!/usr/bin/env python3
import os
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from datasets import TokenizerInterface

class MultiUNDataset(Dataset):
    def __init__(self, 
                 root_dir: str, 
                 tokenizer: TokenizerInterface, 
                 max_length: int,
                 pad_token: int,
                 begin_text_token: int,
                 end_text_token: int):
        assert os.path.isdir(root_dir)
        self.root_dir = os.path.abspath(root_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token = pad_token
        self.begin_text_token = begin_text_token
        self.end_text_token = end_text_token
        self.file_pairs = self._collect_file_pairs()

    def _collect_file_pairs(self) -> list[tuple[str,str]]:
        english_files = sorted([f for f in os.listdir(self.root_dir) if f.endswith('_en.snt')])
        spanish_files = sorted([f for f in os.listdir(self.root_dir) if f.endswith('_es.snt')])
        return list(zip(english_files, spanish_files))

    def __len__(self) -> int:
        return len(self.file_pairs)

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        en_file, es_file = self.file_pairs[idx]

        en_path = os.path.join(self.root_dir, en_file)
        es_path = os.path.join(self.root_dir, es_file)

        with open(en_path, 'r', encoding='utf-8') as f:
            en_text = f.read().strip()

        with open(es_path, 'r', encoding='utf-8') as f:
            es_text = f.read().strip()

        en_tokens = self._tokenize(en_text)
        es_tokens = self._tokenize(es_text)
        padded_en_tokens = F.pad(en_tokens, (0, self.max_length - en_tokens.shape[-1]), value=self.pad_token)
        padded_es_tokens = F.pad(es_tokens, (0, self.max_length - es_tokens.shape[-1]), value=self.pad_token)
        return padded_en_tokens, padded_es_tokens

    def _tokenize(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer.encode(text)[:self.max_length-2] # -2 for begin and end of text tokens
        return torch.tensor([self.begin_text_token] + tokens + [self.end_text_token])