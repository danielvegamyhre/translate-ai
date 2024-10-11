#!/usr/bin/env python3

from typing import Protocol

class TokenizerInterface(Protocol):
    def encode(self, text: str) -> list[int]:
        pass