import logging
from collections import defaultdict
import string
from typing import Iterator
from cs336_basics.bpe_tokenizer_training import SPECIAL_TOKENS, PAT, merge_tokens 
from abc import ABC, abstractmethod
import pickle
import regex as re


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
        if special_tokens:
            self.special_tokens = special_tokens
        else:
            self.special_tokens = SPECIAL_TOKENS

    def encode(self, text: str) -> list[int]:
        pattern = "|".join(re.escape(tok) for tok in self.special_tokens)
        splitted_chunks = re.split(pattern, text)
        # Implement BPE encoding logic here
        pretokens = []
        for splitted_chunk in splitted_chunks:
            # counts_words += Counter(splitted_chunk.split(" "))
            words = re.findall(PAT, splitted_chunk)
            for word in words:
                word_bytes = word.encode("utf-8")
                word_tokens = []
                for i in range(len(word_bytes)):
                    word_tokens.append(self.reverse_vocab[bytes([word_bytes[i]])])
                pretokens.append(word_tokens)
        for i in range(len(pretokens)):
            for pair in self.merges:
                new_indice = self.reverse_vocab[pair[0] + pair[1]]
                pretokens[i] = merge_tokens(self.vocab, pretokens[i], pair, new_indice)
        pretokens_flat = [token for sublist in pretokens for token in sublist]
        return pretokens_flat
        

    def decode(self, indices: list[int]) -> str:
        # Implement BPE decoding logic here
        # print(f"indices: {indices}")
        # print(f"vocab: {self.vocab[indices[0]]}")
        bytes_list = []
        for indice in indices:
            bytes_list.append(self.vocab.get(indice))
        # string = b"".join(bytes_list).decode("utf-8")
        # print(f"bytes_list: {bytes_list}")
        string = b"".join(bytes_list).decode("utf-8", errors="ignore")
        return string
    
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "rb") as f:
            cls.vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            cls.merges = pickle.load(f)
        if special_tokens:
            cls.special_tokens = special_tokens

    @classmethod
    def from_files(cls, BPE_params_filepath, special_tokens=None):
        with open(BPE_params_filepath, "rb") as f:
            BPE_params = pickle.load(f)
            cls.vocab = BPE_params.vocab
            cls.merges = BPE_params.merges
        if special_tokens:
            cls.special_tokens = special_tokens
        else:
            cls.special_tokens = SPECIAL_TOKENS

    def encode_iterable(self, iterable: Iterator[str]) -> Iterator[int]: 
        indices = list(map(int, iterable.encode("utf-8")))
        for pair in self.merges:
            new_indice = 256 + self.reverse_vocab[pair]
            indices = merge_tokens(self.vocab, indices, pair, new_indice)
        yield indices

    

