#!/usr/bin/env python3

def log(msg: str, rank: int = 0):
    if rank == 0:
        print(msg)