import logging
from typing import Iterator
from cs336_basics.bpe_tokenizer_training import SPECIAL_TOKENS, PAT, merge_tokens
from abc import ABC, abstractmethod
import pickle
import regex as re
import json


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


class Tokenizer(ABC):

    @abstractmethod
    def encode(self, string: str) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def decode(self, indices: list[int]) -> str:
        raise NotImplementedError


class BPE_Tokenizer(Tokenizer):
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.special_tokens = special_tokens if special_tokens is not None else []
        # Cache merged token IDs so we don't recompute on every encode() call
        self._merge_ids = [self.reverse_vocab[p[0] + p[1]] for p in self.merges]

    def encode(self, text: str) -> list[int]:
        token_ids = []

        if self.special_tokens:
            # Sort longest-first so overlapping tokens (e.g. "aa" before "a") match correctly
            sorted_special = sorted(self.special_tokens, key=len, reverse=True)
            pattern = "|".join(re.escape(tok) for tok in sorted_special)
            # Capturing group keeps the delimiters in the split result
            chunks = re.split(f"({pattern})", text)
        else:
            chunks = [text]

        special_token_set = set(self.special_tokens)

        for chunk in chunks:
            if not chunk:
                continue
            if chunk in special_token_set:
                # Emit the special token as a single token ID
                token_ids.append(self.reverse_vocab[chunk.encode("utf-8")])
            else:
                # Regular text: pre-tokenise with the GPT-2 regex, then apply BPE merges
                pretokens = []
                for word in re.findall(PAT, chunk):
                    word_bytes = word.encode("utf-8")
                    word_tokens = [self.reverse_vocab[bytes([b])] for b in word_bytes]
                    pretokens.append(word_tokens)
                for i in range(len(pretokens)):
                    for pair, new_id in zip(self.merges, self._merge_ids):
                        pretokens[i] = merge_tokens(self.vocab, pretokens[i], pair, new_id)
                for sublist in pretokens:
                    token_ids.extend(sublist)

        return token_ids

    def decode(self, indices: list[int]) -> str:
        return b"".join(self.vocab[i] for i in indices).decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterator[str]) -> Iterator[int]:
        for line in iterable:
            yield from self.encode(line)

    @classmethod
    def from_files(cls, BPE_params_filepath, special_tokens=None):
        with open(BPE_params_filepath, "rb") as f:
            BPE_params = pickle.load(f)
        return cls(BPE_params.vocab, BPE_params.merges, special_tokens)
    

    @classmethod
    def from_files_text(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, encoding="utf-8") as f:
            vocab = json.load(f)
        with open(merges_filepath, encoding="utf-8") as f:
            merges = [tuple(line.rstrip().split(" ")) for line in f]
        return cls(vocab, merges, special_tokens)

    

